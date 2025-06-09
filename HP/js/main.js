document.addEventListener('DOMContentLoaded', () => {
  // Toggle mobile menu visibility
  document.querySelector('.mobile-menu')
    .addEventListener('click', () => {
      document.querySelector('.nav-right').classList.toggle('show');
    });

  // Image load failure fallback: replace with alt text container
  document.querySelectorAll('img').forEach(img => {
    img.onerror = () => {
      const fallback = document.createElement('div');
      fallback.className   = 'img-fallback';
      fallback.textContent = img.alt || 'Image';
      img.replaceWith(fallback);
      console.error('Failed to load image:', img.src);
    };
  });

  // Fix navigation position on scroll
  const nav = document.querySelector('nav');
  function onScroll() {
    nav.classList.toggle('scrolled', window.scrollY > 50);
  }
  window.addEventListener('scroll', onScroll);
  onScroll();
});

// Optional: log page performance metrics
if ('performance' in window) {
  window.addEventListener('load', () => {
    setTimeout(() => {
      const navEntry = performance.getEntriesByType('navigation')[0];
      console.log('Page load time:', navEntry.loadEventEnd - navEntry.fetchStart, 'ms');
    }, 0);
  });
}
