document.addEventListener('DOMContentLoaded', () => {
    const showBtn = document.getElementById('showBtn');
    const textarea = document.getElementById('htmlContent');
  
    showBtn.addEventListener('click', () => {
      // Disable the button to prevent double clicks
      showBtn.disabled = true;
      showBtn.textContent = 'Loading…';
  
      // Get the active tab…
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tabId = tabs[0].id;
        // …and ask its content script for the HTML
        chrome.tabs.sendMessage(tabId, 'GET_PAGE_HTML', (response) => {
          if (chrome.runtime.lastError) {
            textarea.value = 'Error: ' + chrome.runtime.lastError.message;
          } else {
            textarea.value = response;
          }
          textarea.style.display = 'block';
          showBtn.style.display = 'none';
        });
      });
    });
  });
  