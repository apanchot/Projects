function syntheticResize() {
  var evt = window.document.createEvent('UIEvents'); 
  evt.initUIEvent('resize', true, false, window, 0); 
  window.dispatchEvent(evt);
}

window.addEventListener("scroll", syntheticResize);