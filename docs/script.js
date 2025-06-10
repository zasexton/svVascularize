// Basic interactivity for both index.html and install.html
// – Highlights the current nav link (scroll spy)
// – Adds a shadow / darker background to the nav when scrolling

(function () {
  const nav = document.querySelector('.topnav');
  if (!nav) return;

  // Scroll‑spy: highlight the link whose section is currently in view
  const links = nav.querySelectorAll('nav a[href^="#"]');
  const sections = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);

  function onScroll() {
    const scrollPos = window.scrollY + 120; // offset for sticky nav

    // Toggle shadow + darker BG on nav
    nav.classList.toggle('scrolled', window.scrollY > 10);

    // Determine current section
    let current = sections[0];
    for (const section of sections) {
      if (scrollPos >= section.offsetTop) current = section;
    }

    // Highlight the corresponding link
    links.forEach(l => l.classList.remove('active'));
    const active = nav.querySelector(`nav a[href="#${current.id}"]`);
    if (active) active.classList.add('active');
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  document.addEventListener('DOMContentLoaded', onScroll);
})();


/* ── Copy-to-clipboard for <pre data-copy> blocks ── */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('pre[data-copy]')
    .forEach(block => block.addEventListener('click', () => {
      navigator.clipboard.writeText(block.innerText.trim())
        .then(() => {
          block.classList.add('copied');
          setTimeout(() => block.classList.remove('copied'), 1200);
        });
    }));
});

/* ── Lazy-load Prism.js if any <code class="language-python"> exists ── */
(function () {
  if (!document.querySelector('code.language-python')) return; // nothing to highlight

  // 1) inject Prism core + theme
  const core = document.createElement('script');
  core.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js';
  core.defer = true;

  const theme = document.createElement('link');
  theme.rel = 'stylesheet';
  theme.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css';

  document.head.appendChild(theme);
  document.head.appendChild(core);

  // 2) once core loads, add Python component then highlight
  core.onload = () => {
    const py = document.createElement('script');
    py.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js';
    py.defer = true;
    py.onload = () => window.Prism && Prism.highlightAll();
    document.head.appendChild(py);
  };
})();
