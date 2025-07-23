chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'GET_PAGE_TEXT') {
    sendResponse({ text: document.body.innerText });
  }
});
