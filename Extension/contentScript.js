chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message === 'GET_PAGE_HTML') {
    sendResponse(document.documentElement.outerHTML);
  }
  // returning true isn’t needed here since we sendResponse synchronously
});
