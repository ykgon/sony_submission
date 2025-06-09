(function(){
  // Initialize Google Analytics
  window.dataLayer = window.dataLayer || [];
  function gtag(){ dataLayer.push(arguments); }
  gtag('js', new Date());
  gtag('config', 'G-57VM7KFMC1');

  // Track clicks on the "Register" buttons
  function trackRegisterClick(location) {
    gtag('event', 'register_button_click', {
      button_location: location,
      timestamp: new Date().toISOString()
    });
    console.log(`Register button clicked: ${location}`);
  }

  // Track clicks on "More Info" buttons
  function trackMoreInfoClick(location) {
    gtag('event', 'more_info_button_click', {
      button_location: location,
      timestamp: new Date().toISOString()
    });
    console.log(`More Info clicked: ${location}`);
  }

  // Track when the venue map is enlarged
  function trackVenueMapClick() {
    gtag('event', 'venue_map_enlarge', {
      interaction_type: 'image_modal_open',
      timestamp: new Date().toISOString()
    });
    console.log('Venue map enlarged');
  }

  // Track clicks on the App Store badge
  function trackAppStoreClick() {
    gtag('event', 'app_store_click', {
      app_name: 'shuhari_berlin',
      platform: 'ios',
      source: 'festival_website',
      timestamp: new Date().toISOString()
    });
    console.log('App Store clicked');
  }

  // Track clicks within the Teeraum section
  function trackTeeraumClick(type) {
    gtag('event', 'teeraum_link_click', {
      link_type: type,
      section: 'food_drinks',
      destination: type === 'menu' ? 'speisekarte' : 'teeraum',
      timestamp: new Date().toISOString()
    });
    console.log(`Teeraum link clicked: ${type}`);
  }

  // Track clicks on menu images
  function trackMenuClick(type) {
    gtag('event', 'menu_image_click', {
      menu_type: type,
      section: 'food_drinks',
      interaction_type: 'image_modal_open',
      timestamp: new Date().toISOString()
    });
    console.log(`Menu image clicked: ${type}`);
  }

  // Expose functions globally
  window.trackRegisterClick   = trackRegisterClick;
  window.trackMoreInfoClick   = trackMoreInfoClick;
  window.trackVenueMapClick   = trackVenueMapClick;
  window.trackAppStoreClick   = trackAppStoreClick;
  window.trackTeeraumClick    = trackTeeraumClick;
  window.trackMenuClick       = trackMenuClick;
})();
