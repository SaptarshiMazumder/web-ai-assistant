

// Get elements
const chatDiv = document.getElementById("chat")!;
const askBtn = document.getElementById("ask-btn")!;
const questionInput = document.getElementById("question")! as HTMLInputElement;

// Util: get visible page text from the current active tab
function getPageTextFromActiveTab(): Promise<string> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id !== undefined) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "GET_PAGE_TEXT" },
          (resp) => {
            resolve(resp?.text || "");
          }
        );
      } else {
        resolve("");
      }
    });
  });
}

// Renders markdown with code highlight
function appendMessage(text: string, sender: 'user' | 'bot') {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  bubble.textContent = text;
  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

askBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, "user");

  // Get page text from content script
  const pageText = await getPageTextFromActiveTab();

  // Debug: See if you’re actually getting the text!
  // console.log("Page text length:", pageText.length);

  // Send question + page text to backend
  const resp = await fetch("http://localhost:5000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: pageText, question }),
  });
  const data = await resp.json();

  appendMessage(data.answer, "bot");
  questionInput.value = "";
};
