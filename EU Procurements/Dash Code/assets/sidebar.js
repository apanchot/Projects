myID = document.getElementsByClassName("sidebar");
contentID = document.getElementsByClassName("content");

codeID = document.getElementsByClassName("button_code");
countryID = document.getElementsByClassName("button_country");
businessID = document.getElementsByClassName("button_business");


var myScrollFunc = function() {
  var y = window.scrollY;
  pathname = window.location.pathname;
  if (y >= 600) {
    myID.sidebar.style.visibility = "visible";
    myID.sidebar.style.width = "20%";
    contentID.content.style.marginLeft = "20%";
    if (pathname == '/codes') {
    codeID.button_code.style.visibility = "visible";
    countryID.button_country.style.visibility = "collapse";
    businessID.button_business.style.visibility = "collapse";
    }
    else if (pathname == '/countries') {
    countryID.button_country.style.visibility = "visible";
    codeID.button_code.style.visibility = "collapse";
    businessID.button_business.style.visibility = "collapse";
    }
    else if (pathname == '/businesses') {
    businessID.button_business.style.visibility = "visible";
    countryID.button_country.style.visibility = "collapse";
    codeID.button_code.style.visibility = "collapse";
    }
    else {
    countryID.button_country.style.visibility = "collapse";
    codeID.button_code.style.visibility = "collapse";
    businessID.button_business.style.visibility = "collapse";
    };
  }
  else {
    myID.sidebar.style.visibility = "collapse"; 
    myID.sidebar.style.width = "0%";
    contentID.content.style.marginLeft = "0%";
    countryID.button_country.style.visibility = "collapse";
    codeID.button_code.style.visibility = "collapse";
    businessID.button_business.style.visibility = "collapse";
  }
};

window.addEventListener("scroll", myScrollFunc);