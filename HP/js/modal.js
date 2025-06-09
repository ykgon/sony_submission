(function(){
  const modal    = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');

  // Trap focus inside the modal for accessibility
  function trapFocus(modalEl) {
    const focusable = modalEl.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const first = focusable[0];
    const last  = focusable[focusable.length - 1];

    function onTab(e) {
      if (e.key !== 'Tab') return;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    modalEl.addEventListener('keydown', onTab);
  }

  // Open image modal and optionally track
  function openImageModal(imgEl, trackingType = 'venue_map') {
    modalImg.src = imgEl.src;
    modalImg.alt = imgEl.alt;
    modal.classList.add('show');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    trapFocus(modal);

    // Only track venue map here; menu and tea tracking handled elsewhere
    if (trackingType === 'venue_map') {
      window.trackVenueMapClick();
    }
  }

  // Close the image modal
  function closeImageModal() {
    modal.classList.remove('show');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    document.querySelector('.venue-map-img')?.focus();
  }

  // Handle keyboard activation (Enter/Space) on images
  function handleImageKeyDown(e, imgEl) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    const src = imgEl.src;
    let type = 'venue_map';
    if (src.includes('ICHIJU-SANSAI_menu')) type = 'ichiju-sansai';
    else if (src.includes('Teeraum_menu'))    type = 'teeraum';

    openImageModal(imgEl, type);
    if (type !== 'venue_map') window.trackMenuClick(type);
  }

  // Close modal on Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal.classList.contains('show')) {
      closeImageModal();
    }
  });

  // Expose modal controls globally
  window.openImageModal     = openImageModal;
  window.closeImageModal    = closeImageModal;
  window.handleImageKeyDown = handleImageKeyDown;
})();
