---
title: "The First Incidient Made Negative Impact to Business"
date: 2026-06-20T21:46:48+07:00
draft: false
description: "My first production incident that hurt the business — what went wrong, the behaviors and assumptions that led to it, and the practical lessons I now follow to prevent it from happening again."
categories: ["Engineering", "Incident"]
tags: ["ClickHouse", "Postmortem", "Performance", "Lessons Learned"]
---

- [The day a "small" feature took down search](#the-day-a-small-feature-took-down-search)
- [What I shipped](#what-i-shipped)
- [What went wrong](#what-went-wrong)
- [The mistake that cost me the most time: rolling back](#the-mistake-that-cost-me-the-most-time-rolling-back)
- [The real root cause](#the-real-root-cause)
  - [The query couldn't use the index](#the-query-couldnt-use-the-index)
  - ["Not indexed yet" is the worst case](#not-indexed-yet-is-the-worst-case)
  - [Why 1.5 rps was fine but 9 rps fell over](#why-15-rps-was-fine-but-9-rps-fell-over)
- [How I stopped the bleeding](#how-i-stopped-the-bleeding)
- [The proper fix](#the-proper-fix)
- [What I actually learned](#what-i-actually-learned)
  - [The behaviors I'm changing](#the-behaviors-im-changing)

## The day a "small" feature took down search

Every engineer has a first one. The first time your change is the reason a graph turns red, the reason a teammate pings you with *"is something wrong with the service?"*, the reason the business feels it. This is mine.

It wasn't a dramatic, clever bug. It was a single, innocent-looking database query that was completely fine in code review, completely fine in testing, and completely fine in production — right up until enough people used it at the same time. That last part is the whole story.

I'm writing it down because the technical lesson is useful, but the *behavioral* lesson is the one I never want to relearn.

## What I shipped

I deployed the face recognition service. That's critical service in our company in production. All of traffic should be passed here before doing anything else.

After 6 month release, everything work correctly. We just mantained and do some improvement and I'm also the person in charge for this.

One of the feature that I shiped relevant to improve a 1:N search API, and the task was to add an **optional indexing step**: when a flag is enabled, the embedding we generate gets inserted into our vector store (Qdrant) and our metadata store (ClickHouse). To keep the search response fast, I moved that indexing work into a **background task** so it wouldn't block the response.

As part of indexing, I added what felt like a responsible, defensive touch: before inserting, **check ClickHouse whether this `image_path` is already indexed**, so we don't store duplicates. Idempotency. Good hygiene. What could go wrong?

```sql
SELECT image_path FROM biometric_images
WHERE image_path IN ('path1', 'path2', ...)
```

## What went wrong

Shortly after deploy, latency started climbing under real traffic. The latency is unstable. Unfortunately, the service that call my service have retried logic and it make the RPS reached at around **9 requests per second**, ClickHouse queries began timing out. And because our **search** path shares the *same* ClickHouse instance, those timeouts spilled over and degraded search too. A background dedup check I added for indexing was now taking down the user-facing search endpoint.

The strange part: at **~1.5 rps**, the exact same code was perfectly healthy. No errors, normal latency. The problem only existed under load.

## The mistake that cost me the most time: rolling back

My first instinct was the textbook one: *the new feature is slow, roll it back.*

So I reverted the 1:N change. And it **didn't help.** The service stayed unstable.

That was the moment of real discomfort — the "undo" button didn't undo the problem. In hindsight, the rollback failing was the most valuable signal of the entire incident, because it told me my mental model was wrong. The slowness wasn't in the *feature logic* I had reverted. It was in the **ClickHouse query**, which was still running. I had rolled back the wrapper and left the actual culprit in place.

## The real root cause

### The query couldn't use the index

Here's the table definition:

```sql
ENGINE = MergeTree()
ORDER BY (client_id, image_path)
```

In ClickHouse, the `ORDER BY` key *is* the (sparse) primary index. It can only skip data when your filter uses a **prefix** of that key — here, that prefix starts with `client_id`.

My query filtered on `image_path` **alone**, the second column, with no `client_id`. So the index was useless, and ClickHouse had no choice but to **scan the entire table on every single request**. For this query, `image_path` was effectively a non-indexed column.

### "Not indexed yet" is the worst case

It gets worse. Proving a row is *absent* has no early exit — ClickHouse has to look everywhere before it can confidently say "not found." And since fresh batches are mostly *new* paths, my dedup check was doing its single most expensive operation — a full scan that returns nothing — on nearly every request.

### Why 1.5 rps was fine but 9 rps fell over

A single full scan is cheap enough to be invisible. The cost doesn't add up linearly under concurrency — it compounds:

- **CPU oversubscription** — each scan grabs up to `max_threads` (roughly the core count). One scan uses the cores well; nine concurrent scans demand many times the cores that exist, and the machine starts thrashing on context switches.
- **I/O contention** — full scans read the whole table, so concurrent scans fight over the same disk and page-cache bandwidth, slowing each other down.
- **Multiplicative latency** — a scan that takes ~0.7s alone stays ~0.7s at 1.5 rps (little overlap, under the timeout). At 9 rps, nine of them run at once, each starved for CPU and I/O, and effective latency balloons past the timeout. Then requests start failing.

The data didn't change between 1.5 and 9 rps. Only the **concurrency multiplier** did. 1.5 rps sat just under the cliff. 9 rps went over it.

## How I stopped the bleeding

The mitigation was to **remove the existing-path check** entirely and insert rows directly. That took the full scan out of the hot path and immediately restored stability under load.

It's not free: I gave up write-time deduplication. But `MergeTree` tolerates duplicate rows, so it's an acceptable short-term trade — stability now, correct dedup later.

## The proper fix

To get dedup back *without* a full scan, in rough order of effort:

1. **Put `client_id` back in the filter** — `WHERE client_id IN (...) AND image_path IN (...)`. This engages the `(client_id, image_path)` prefix index and skips almost all the data. Cheapest fix if the batch already knows its client IDs.
2. **Add a data-skipping index on `image_path`** — e.g. a `bloom_filter` index. This lets "does this path exist?" lookups skip granules cheaply, which is exactly the access pattern here.
3. **Switch to `ReplacingMergeTree`** keyed on the path so duplicates collapse during background merges — no read-before-write at all, at the cost of `FINAL`/query-time dedup and eventual consistency.

Options 1 and 2 keep correctness while killing the scan, with the least architectural change.

## What I actually learned

The ClickHouse lesson is real — *treat your OLAP store like an OLAP store; never filter the hot path on a non-prefix, non-indexed column.* But the lessons that stuck with me are about how I worked, not what I typed.

**A query that is fine alone can be a fire under concurrency.** "It's fast in testing" only means "it's fast at the concurrency I tested." Performance is a function of load, and I had only ever measured it at a load of one.

**Shared infrastructure means blast radius.** I thought I was changing the *indexing* path. I was actually putting pressure on a database that *search* depends on. The boundary I drew in my head ("this is just the background task") didn't exist in the system.

**When the rollback doesn't fix it, your model is wrong — listen to that.** I almost kept reverting more things. The faster move was to stop and ask *what is actually still executing?*

### The behaviors I'm changing

- **Load-test new hot-path DB queries at realistic concurrency** (for us, ≥ 9 rps), not just functionally. Regressions like this are invisible at rps = 1.
- **`EXPLAIN` every new ClickHouse query in review** and check rows-scanned. "Does it use the index?" is now a question I ask before merge, not after an incident.
- **Treat 'defensive' read-before-write as a real cost**, especially on columnar stores. A dedup check that scans a table is not free hygiene — it's a full scan wearing a nice outfit.
- **Map shared dependencies before shipping.** If my change touches a resource another critical path uses, that path is in my blast radius whether I like it or not.
- **Trust the rollback signal.** If undoing the change doesn't undo the symptom, stop changing things and re-derive the root cause.

No one enjoys being the cause of a red graph. But this one taught me more than a dozen smooth deploys did — and now it's written down, so the next person (probably future me) gets the lesson without the incident.


