// At the top of contents.ts
declare global {
  interface Window {
    find(
      string: string,
      caseSensitive?: boolean,
      backwards?: boolean,
      wrap?: boolean,
      wholeWord?: boolean,
      searchInFrames?: boolean,
      showDialog?: boolean
    ): boolean;
  }
}

chrome.runtime.onMessage.addListener(
  (
    req: { type: string; excerpt?: string },
    sender,
    sendResp
  ) => {
    if (req.type === 'GET_PAGE_TEXT') {
      sendResp?.({ text: document.body.innerText });
    }

    if (req.type === 'JUMP_TO_POSITION') {
      const excerpt = req.excerpt || '';
      if (!excerpt) return;

      let found = false;
      let toTry = [
        excerpt,
        excerpt.trim(),
        excerpt.slice(0, 60),
        excerpt.split(' ').slice(0, 6).join(' '),
        excerpt.split(' ').slice(0, 10).join(' '),
      ];

      for (const snippet of toTry) {
        if (!snippet) continue;
        if (window.find(snippet, false, false, true, false, false, false)) {
          found = true;
          break;
        }
      }

      if (!found) {
        alert('Could not locate the answer in the page.');
      }
    }
  }
);

export {}; // <--- needed if you use declare global in a module file
