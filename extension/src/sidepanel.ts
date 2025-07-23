import { marked } from "marked";
import hljs from "highlight.js";

const chatDiv = document.getElementById("chat")!;
const askBtn = document.getElementById("ask-btn")!;
const questionInput = document.getElementById("question")! as HTMLInputElement;

// --- New: Get all page data (text, tables, links, images) ---
function getPageDataFromActiveTab(): Promise<{
  text: string;
  tables: string[];
  links: Array<{ text: string; href: string }>;
  images: Array<{ alt: string; src: string }>;
}> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id !== undefined) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "GET_PAGE_DATA" },
          (resp) => {
            resolve(resp || { text: "", tables: [], links: [], images: [] });
          }
        );
      } else {
        resolve({ text: "", tables: [], links: [], images: [] });
      }
    });
  });
}

// --- Chat message rendering (with markdown for bot) ---
function appendMessage(text: string, sender: 'user' | 'bot' | 'thinking'): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;

  if (sender === 'bot') {
    const parsed = marked.parse(text);
    if (parsed instanceof Promise) {
      parsed.then((html) => {
        bubble.innerHTML = html;
        bubble.querySelectorAll("pre code").forEach((block) => {
          hljs.highlightElement(block as HTMLElement); // code highlighting
        });
      });
    } else {
      bubble.innerHTML = parsed;
      bubble.querySelectorAll("pre code").forEach((block) => {
        hljs.highlightElement(block as HTMLElement); // code highlighting
      });
    }
  } else {
    bubble.textContent = text;
  }

  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return bubble;
}

// --- Source link logic ---
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

// --- Main ask button handler ---
askBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, "user");

  const thinkingBubble = appendMessage("Thinking...", "thinking");

  // Get rich page data (not just text)
  const pageData = await getPageDataFromActiveTab();

  // Compose a context string for the LLM (text + tables + links + images)
  let context = pageData.text;

  if (pageData.tables && pageData.tables.length > 0) {
    context += "\n\n# Tables:\n";
    context += pageData.tables.map((t, i) => `Table ${i+1}:\n${t}`).join("\n\n");
  }

  if (pageData.links && pageData.links.length > 0) {
    context += "\n\n# Links:\n";
    context += pageData.links.map((l, i) => `- [${l.text}](${l.href})`).join("\n");
  }

  if (pageData.images && pageData.images.length > 0) {
    context += "\n\n# Images:\n";
    context += pageData.images.map((img, i) => `- alt: "${img.alt}" src: ${img.src}`).join("\n");
  }

  try {
    const resp = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: context, question }),
    });
    const data = await resp.json();

    thinkingBubble.remove();

    appendMessage(data.answer, "bot");
    if (data.sources && data.sources.length > 0) {
      renderSources(data.sources);
    }
  } catch (err) {
    thinkingBubble.textContent = "Error. Please try again.";
  }

  questionInput.value = "";
};
