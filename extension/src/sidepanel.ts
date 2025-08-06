import { marked } from "marked";
import hljs from "highlight.js";
import { crawlEntireSite, debugLog } from "./siteCrawler";


// crawl webpage start

const BACKEND_BASE_URL = "http://localhost:5000";

function ensureContentScript(tabId: number, cb: () => void) {
  debugLog(`Ensuring content script is injected for tab ${tabId}`);
  chrome.tabs.sendMessage(tabId, { type: "PING" }, (response) => {
    if (chrome.runtime.lastError) {
      // Not present, inject content script
      chrome.scripting.executeScript(
        {
          target: { tabId },
          files: ["dist/content.js"], // or "content.bundle.js" if you use webpack
        },
        () => {
          // Wait a tick for it to initialize
          setTimeout(cb, 100);
        }
      );
    } else {
      cb();
    }
  });
}

function renderUsefulLinks(links: {text: string, href: string}[]) {
    // Remove previous block if you want only one set of links at a time
    const prev = document.getElementById('useful-links-block');
    if (prev) prev.remove();

    if (!links || !links.length) return;
    const chatDiv = document.getElementById("chat")!;
    const block = document.createElement('div');
    block.id = 'useful-links-block';
    block.className = 'system-message';
    block.style.margin = '16px 0';
    block.innerHTML = `<b>You may find these links useful:</b><ul style="margin-top:4px;margin-bottom:4px;padding-left:16px;">
        ${links.map(link => `<li><a href="${link.href}" target="_blank" rel="noopener noreferrer">${link.text || link.href}</a></li>`).join('')}
    </ul>`;
    chatDiv.appendChild(block);
    block.scrollIntoView({behavior: "smooth"});
}


const chatDiv = document.getElementById("chat")!;
const questionInput = document.getElementById("question")! as HTMLInputElement;

// --- SmartQA log streaming ---
let smartqaLogSocket: WebSocket | null = null;

function renderLLMLinksMessage(llmMessage: string, links: {text: string, href: string}[]) {
    const prev = document.getElementById('llm-links-block');
    if (prev) prev.remove();

    const chatDiv = document.getElementById("chat")!;
    const block = document.createElement('div');
    block.id = 'llm-links-block';
    block.className = 'system-message';
    block.style.margin = '16px 0';
    block.innerHTML = `
        <div style="margin-bottom:6px;">${llmMessage}</div>
        <ul style="margin-top:4px;margin-bottom:4px;padding-left:16px;">
            ${
                links.map(link =>
                    `<li>
                        <a href="${link.href}" target="_blank" rel="noopener noreferrer">
                            ${link.text || link.href}
                        </a>
                        <br>
                        <small style="color:#888;">${link.href}</small>
                    </li>`
                ).join('')
            }
        </ul>`;
    chatDiv.appendChild(block);
    block.scrollIntoView({behavior: "smooth"});
}



function connectSmartQALogSocket(logContainer: HTMLElement) {
  if (smartqaLogSocket) {
    smartqaLogSocket.close();
  }
  smartqaLogSocket = new WebSocket("ws://localhost:5000/ws/smartqa-logs");
smartqaLogSocket.onmessage = (event) => {
    let isJSON = false;
    let msg: any;
    try {
        msg = JSON.parse(event.data);
        isJSON = true;
    } catch (e) {}

    // 1. LLM links (unchanged)
    if (isJSON && msg && msg.type === "llm_links_message") {
        renderLLMLinksMessage(msg.message, msg.links);
        return;
    }
    if (isJSON && msg && msg.type === "selected_links") {
        renderUsefulLinks(msg.links);
        return;
    }

    // 3. All other logs: append as a <div> (NOT with innerText)
    const logLine = document.createElement("div");
    logLine.textContent = event.data;
    logContainer.appendChild(logLine);
    logContainer.scrollTop = logContainer.scrollHeight;
};


  smartqaLogSocket.onclose = () => {
    // Optionally reconnect or clear
  };
}

// --- DROPDOWN to toggle between Ask and Ask Smart ---

const dropdown = document.createElement("select");
dropdown.id = "ask-mode";
dropdown.style.marginLeft = "8px";

const optionAsk = document.createElement("option");
optionAsk.value = "ask";
optionAsk.textContent = "Ask";
dropdown.appendChild(optionAsk);

const optionSmart = document.createElement("option");
optionSmart.value = "smart";
optionSmart.textContent = "Ask Smart";
dropdown.appendChild(optionSmart);

const optionGemini = document.createElement("option");
optionGemini.value = "gemini";
optionGemini.textContent = "Ask Gemini";
dropdown.appendChild(optionGemini);

// --- Smart Ask button (always visible, no dropdown) ---
const smartBtn = document.createElement("button");
smartBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="#fff"/></svg>`;
smartBtn.title = "Send";
smartBtn.id = "smart-ask-btn";
smartBtn.style.marginLeft = "8px";
smartBtn.style.display = "inline-block";

const inputRow = document.getElementById("inputRow");
if (inputRow) {
  inputRow.appendChild(dropdown);
  inputRow.appendChild(smartBtn);
}





// Existing: get current page data
function getPageDataFromActiveTab(): Promise<{
  text: string;
  tables: string[];
  links: Array<{ text: string; href: string }>;
  images: Array<{ alt: string; src: string }>;
}> {
  console.log("getting page data from active tab");
  return new Promise((resolve) => {
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const tabId = tabs[0]?.id;
  if (typeof tabId === "number") {
    ensureContentScript(tabId, () => {
      chrome.tabs.sendMessage(
        tabId,
        { type: "GET_PAGE_DATA" },
        (resp) => {
          resolve(resp || { text: "", tables: [], links: [], images: [] });
        }
      );
    });
  } else {
    resolve({ text: "", tables: [], links: [], images: [] });
  }
});
  });
}

// Markdown rendering for chat bubbles (fix: no async needed)
function appendMessage(text: string, sender: 'user' | 'bot' | 'thinking'): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;

  if (sender === 'thinking') {
    const thinkingText = document.createElement('span');
    thinkingText.textContent = text;

    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = 'Show/Hide Thinking';
    toggleBtn.style.marginLeft = '10px';
    toggleBtn.style.cursor = 'pointer';

    const logContainer = document.createElement('div');
    logContainer.className = 'thinking-log';
    logContainer.style.display = 'none';
    logContainer.style.whiteSpace = 'pre-wrap';
    logContainer.style.maxHeight = '200px';
    logContainer.style.overflowY = 'auto';
    logContainer.style.marginTop = '5px';
    logContainer.style.padding = '5px';
    logContainer.style.border = '1px solid #ccc';
    logContainer.style.borderRadius = '4px';


    toggleBtn.onclick = () => {
      const isHidden = logContainer.style.display === 'none';
      logContainer.style.display = isHidden ? 'block' : 'none';
    };

    bubble.appendChild(thinkingText);
    bubble.appendChild(toggleBtn);
    bubble.appendChild(logContainer);

    // Store references for later updates
    (bubble as any).thinkingText = thinkingText;
    (bubble as any).logContainer = logContainer;

  } else if (sender === 'bot') {
    const parsed = marked.parse(text);
    if (parsed instanceof Promise) {
      parsed.then(html => {
        bubble.innerHTML = html;
        bubble.querySelectorAll("pre code").forEach((block) => {
          hljs.highlightElement(block as HTMLElement);
        });
      });
    } else {
      bubble.innerHTML = parsed;
      bubble.querySelectorAll("pre code").forEach((block) => {
        hljs.highlightElement(block as HTMLElement);
      });
    }
  } else {
    bubble.textContent = text;
  }
  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return bubble;
}

// Source link logic (unchanged)
function renderSources(sources: Array<{ excerpt: string; title?: string; url?: string }>) {
  if (!sources || sources.length === 0) return;

  const srcDiv = document.createElement("div");
  srcDiv.className = "sources";
  srcDiv.innerHTML = "<b style='color:#444;margin-bottom:2px;'>Sources:</b>";

  sources.forEach((src) => {
    const infoText =
      (src.title ? `<b>${src.title}</b>` : "") +
      (src.url ? ` <span style="color:#0a5; font-size:0.93em;">${src.url}</span>` : "");

    // Show excerpt as quote, then source info
    const srcBlock = document.createElement("div");
    srcBlock.style.marginBottom = "8px";
    srcBlock.innerHTML =
      `<div style="font-size:0.98em; color:#555; margin-bottom:1px;"><i>${src.excerpt}</i></div>` +
      `<div>${infoText}</div>`;

    srcDiv.appendChild(srcBlock);
  });

  chatDiv.appendChild(srcDiv);
}

// function jumpToSource(excerpt: string) {
//   chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
//     if (tabs[0]?.id !== undefined) {
//       chrome.tabs.sendMessage(
//         tabs[0].id,
//         { type: "JUMP_TO_POSITION", excerpt },
//         () => {}
//       );
//     }
//   });
// }

// --- Smart Ask button logic ---
smartBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;

  // Get selected tool
  const selectedTool = dropdown.value;
  console.log('Selected tool:', selectedTool);

  // Add "Smart QA" or "Gemini QA" tag above the bubble
  const tag = document.createElement('div');
  tag.textContent = selectedTool === 'gemini' ? "Gemini QA" : "Smart QA";
  tag.style.fontSize = "0.8em";
  tag.style.color = selectedTool === 'gemini' ? "#0af" : "#0a5";
  tag.style.fontWeight = "bold";
  tag.style.letterSpacing = "0.03em";
  tag.style.margin = "0 10px 0 0";
  tag.style.textAlign = "right";
  tag.style.width = "100%";
  chatDiv.appendChild(tag);
  appendMessage(question, "user");
  const thinkingBubble = appendMessage("Thinking...", "thinking");
  const logContainer = (thinkingBubble as any).logContainer;
  logContainer.innerText = "";
  if (selectedTool === 'gemini') {
    // No log socket for Gemini
  } else {
    connectSmartQALogSocket(logContainer);
  }

  // --- NEW: Always use getPageDataFromActiveTab to ensure injection ---
  const pageData = await getPageDataFromActiveTab();
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const tab = tabs[0];
    const page_url = tab?.url ?? "";
    const body = {
      text: pageData.text,
      question,
      links: pageData.links,
      page_url,
    };
    try {
      let endpoint = "http://localhost:5000/ask-smart";
      if (selectedTool === 'gemini') {
        endpoint = "http://localhost:5000/ask-gemini";
      }
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      (thinkingBubble as any).thinkingText.textContent = 'Completed thinking.';
      if (smartqaLogSocket && selectedTool !== 'gemini') {
        smartqaLogSocket.close();
      }

      // Show the answer
      appendMessage(data.answer, "bot");
      // Show sources (if any)
      if (data.sources && data.sources.length > 0) {
        renderSources(data.sources);
      }

      // If not sufficient, show LLM-picked links (plain, no style)
      if (data.sufficient === false && data.selected_links && data.selected_links.length > 0) {
        appendMessage("Try checking one of these links for more info:", "bot");
        data.selected_links.forEach((l: any) => {
          appendMessage(`• ${l.text} — ${l.href}`, "bot");
        });
      }

      if (data.visited_urls && data.visited_urls.length > 0) {
        const urls = data.visited_urls;
        let msg = "Pages visited:\n";
        urls.forEach((url: string, idx: number) => {
          msg += `• ${url}\n`;
        });
        msg += `\nYou can also view the last visited page for more details..`;
        appendMessage(msg, "bot");
      }

    } catch (err) {
      (thinkingBubble as any).thinkingText.textContent = selectedTool === 'gemini' ? "Error (Gemini QA). Please try again." : "Error (smart QA). Please try again.";
    }
    questionInput.value = "";
  });
};

// --- DROPDOWN LOGIC: toggle visible button ---
// dropdown.addEventListener("change", () => {
//   if (dropdown.value === "ask") {
//     smartBtn.style.display = "inline-block";
//   } else {
//     smartBtn.style.display = "inline-block";
//   }
// });

// Default to "Ask"
// dropdown.value = "smart";
// smartBtn.style.display = "inline-block";
