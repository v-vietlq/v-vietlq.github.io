(function() {
  // On page load or when changing themes, best to add inline in `head` to avoid FOUC
  const themeToggle = document.querySelector('.darkmode-toggle input');
  const light = 'light';
  const dark = 'dark';
  let isDark = localStorage.theme === dark || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
  if (isDark) {
    document.documentElement.classList.add(dark);
    themeToggle.checked = true;
  } else {
    document.documentElement.classList.remove(dark);
    themeToggle.checked = false;
  }

  themeToggle.addEventListener('change', function () {
    if (this.checked) {
      localStorage.theme = dark;
      document.documentElement.classList.add(dark);
    } else {
      localStorage.theme = light;
      document.documentElement.classList.remove(dark);
    }
  });

  const navbarMenuToggle = document.getElementById('navbar-menu-toggle');
  const navbarMenu = document.getElementById('navbar-menu');
  const navbarLangToggle = document.getElementById('navbar-lang-toggle');
  const navbarLang = document.getElementById('navbar-lang');

  document.addEventListener('click', function (event) {
    const target = event.target;
    if (navbarMenuToggle.contains(target)) {
      navbarLang && navbarLang.classList.add('hidden');
      navbarMenu && navbarMenu.classList.toggle('hidden');
    } else if (navbarLangToggle.contains(target)) {
      navbarMenu && navbarMenu.classList.add('hidden');
      navbarLang && navbarLang.classList.toggle('hidden');
    } else {
      navbarMenu && navbarMenu.classList.add('hidden');
      navbarLang && navbarLang.classList.add('hidden');
    }
  });

  const indicator = document.querySelector('.nav-indicator');
	const items = document.querySelectorAll('.nav-item');

	function handleIndicator(el) {
		items.forEach(item => {
			item.classList.remove('is-active');
			item.removeAttribute('style');
		});

		indicator.style.width = `${el.offsetWidth}px`;
		indicator.style.left = `${el.offsetLeft}px`;
		indicator.style.backgroundColor = el.getAttribute('active-color');

		el.classList.add('is-active');
		el.style.color = el.getAttribute('active-color');
	}


	items.forEach((item, index) => {
		item.addEventListener('click', (e) => { handleIndicator(e.target) });
		item.classList.contains('is-active') && handleIndicator(item);
	});

    // get button element
	const disqusBtn = document.getElementById("disqus-btn");
	const fbBtn = document.getElementById("fb-btn");

	// get message board element
	const disqusBoard = document.getElementById("disqus-comments");
	const fbBoard = document.getElementById("facebook-comments");
	// Add click event listeners to each button

	disqusBtn.addEventListener("click", function () {
		disqusBoard.style.display = "block";
		fbBoard.style.display = "none";
	});
	fbBtn.addEventListener("click", function () {
		disqusBoard.style.display = "none";
		fbBoard.style.display = "block";
	});

})();
