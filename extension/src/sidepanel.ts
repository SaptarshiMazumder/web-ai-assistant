import {marked} from "marked";

import hljs from "highlight.js";


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

async function appendMessage(text: string, sender: 'user' | 'bot' | 'thinking'): Promise<HTMLElement> {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;

  // Bot messages (and optionally 'thinking') get markdown formatting
  if (sender === 'bot') {
    const parsed = await marked.parse(text);
    bubble.innerHTML = parsed;
    // Highlight any code blocks
    bubble.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block as HTMLElement);
    });
  } else {
    bubble.textContent = text;
  }

  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return bubble;
}


// ---- SOURCE LOGIC STARTS HERE ----
function renderSources(sources: Array<{ excerpt: string }>) {
  if (!sources || sources.length === 0) return;

  const srcDiv = document.createElement("div");
  srcDiv.className = "sources";
  srcDiv.innerHTML = "<b style='color:#444;margin-bottom:2px;'>Sources:</b>";

  sources.forEach((src) => {
    const btn = document.createElement("button");
    btn.className = "source-btn";
    btn.title = src.excerpt;
    btn.onclick = () => jumpToSource(src.excerpt);
    btn.textContent = src.excerpt.length > 100 ? src.excerpt.slice(0, 100) + "..." : src.excerpt;
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
  await appendMessage(question, "user");

  const thinkingBubble = await appendMessage("Thinking...", "thinking");

  const pageText = await getPageTextFromActiveTab();

  try {
    const resp = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: pageText, question }),
    });
    const data = await resp.json();

    thinkingBubble.remove();

    await appendMessage(data.answer, "bot");
    if (data.sources && data.sources.length > 0) {
      renderSources(data.sources);
    }
  } catch (err) {
    thinkingBubble.textContent = "Error. Please try again.";
  }

  questionInput.value = "";
};


