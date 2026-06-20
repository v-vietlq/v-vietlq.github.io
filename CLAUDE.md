# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal Hugo blog ("Viet's blog") deployed to GitHub Pages at https://v-vietlq.github.io. Content is Markdown; the site is built with **Hugo extended v0.122.0** and styled by a vendored Tailwind CSS theme.

## Commands

```bash
# Local dev server with drafts + live reload
hugo server -D

# Production build (mirrors CI)
hugo --gc --minify

# Scaffold a new post (page bundle under content/post/<name>/index.md)
hugo new content post/<name>/index.md
```

Requires Hugo **extended** (the theme compiles SCSS/uses Tailwind asset pipeline). CI uses `hugo_extended` 0.122.0 + Dart Sass.

## Deployment

Pushing to `main` triggers `.github/workflows/hugo.yaml`, which builds with `--gc --minify` and deploys `./public` to GitHub Pages. There is no separate publish step — **merging to `main` publishes the live site.** Set `draft: false` in a post's front matter before it can appear in a production build.

## Architecture

- **`config/_default/`** — split TOML config (Hugo merges all files): `hugo.toml` (baseURL, markup/Goldmark math passthrough, taxonomies), `params.toml` (theme params: giscus + Facebook comments, social links, header/footer), `languages.toml` (menu + i18n), `services.toml` (analytics/comments). `config/production/services.toml` overrides for prod builds (e.g. the real Google Analytics ID).
- **`themes/tailwind/`** — the theme, **vendored directly in-tree (not a git submodule)**. It is a modified fork of [tomowang/hugo-theme-tailwind](https://github.com/tomowang/hugo-theme-tailwind) with local additions (Facebook comments, "chat with website" widget). Editing files here changes the live site, but the theme's own `custom_head.html`/`custom_footer.html` say *do not edit directly* — override from the project root instead (see below).
- **`layouts/partials/`** — project-level overrides that win over the theme's same-named files. Current overrides: `head_custom.html`, `footer_custom.html` (injection points for custom HTML before `</head>`/`</body>`), and `math.html` (KaTeX CDN + auto-render). Add site customizations here rather than touching `themes/tailwind/`.
- **`content/post/<slug>/index.md`** — posts are **page bundles**: each post is a directory with `index.md` plus an `images/` subfolder for co-located assets. `content/homepage/` is a headless bundle.

## Conventions specific to this repo

- **Math:** posts using KaTeX must set `math: true` in front matter. Delimiters are configured via Goldmark passthrough in `hugo.toml` — `$...$` inline, `$$...$$` / `\[...\]` block.
- **Front matter gotcha:** check `draft:` values carefully — `draft: fase` (typo) is truthy-as-string and behaves unexpectedly; use `draft: false`.
- **Comments:** giscus (GitHub Discussions) and Facebook comments are both enabled per-page via `params.toml`.
- **Tailwind CSS** is served from the prebuilt `themes/tailwind/assets/css/index.css` via Hugo Pipes. Content/layout edits need no CSS rebuild. Only regenerate CSS if you change Tailwind classes that aren't yet in `index.css` — run the theme's `npm run dev-tailwind` (from `themes/tailwind/`) to recompile `main.css` → `index.css`.
