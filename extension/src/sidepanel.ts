const chatDiv = document.getElementById("chat")!;
const askBtn = document.getElementById("ask-btn")!;
const questionInput = document.getElementById("question")! as HTMLInputElement;

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

function appendMessage(text: string, sender: 'user' | 'bot') {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  bubble.textContent = text;
  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

// ---- SOURCE LOGIC STARTS HERE ----
function renderSources(sources: Array<{ excerpt: string }>) {
  if (!sources || sources.length === 0) return;

  const srcDiv = document.createElement("div");
  srcDiv.className = "sources";
  srcDiv.innerHTML = "<b>Sources:</b> ";

  sources.forEach((src, idx) => {
    const btn = document.createElement("button");
    btn.textContent = `Source ${idx + 1}`;
    btn.style.marginRight = "8px";
    btn.onclick = () => jumpToSource(src.excerpt);
    srcDiv.appendChild(btn);
  });

  chatDiv.appendChild(srcDiv);
}

function jumpToSource(excerpt: string) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id !== undefined) {
      chrome.tabs.sendMessage(
        tabs[0].id,
        { type: "JUMP_TO_POSITION", excerpt },
        () => {}
      );
    }
  });
}
// ---- SOURCE LOGIC ENDS HERE ----

askBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, "user");

  const pageText = await getPageTextFromActiveTab();

  const resp = await fetch("http://localhost:5000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: pageText, question }),
  });
  const data = await resp.json();

  appendMessage(data.answer, "bot");
  if (data.sources && data.sources.length > 0) {
    renderSources(data.sources);
  }
  questionInput.value = "";
};
