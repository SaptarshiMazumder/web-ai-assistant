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

function renderUsefulLinks(links: {text: string, href: string}[], sessionId?: number) {
    // Remove previous block if you want only one set of links at a time
    const prev = document.getElementById('useful-links-block');
    if (prev) prev.remove();

    if (!links || !links.length) return;
    const chatDiv = document.getElementById("chat")!;
    const block = document.createElement('div');
    if (sessionId !== undefined) {
      block.setAttribute('data-session-id', String(sessionId));
    }
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

// --- Website Indexed Status UI ---
const indexStatus = document.createElement('div');
indexStatus.id = 'index-status';
indexStatus.style.display = 'inline-flex';
indexStatus.style.alignItems = 'center';
indexStatus.style.gap = '6px';
indexStatus.style.marginLeft = '8px';
indexStatus.style.fontSize = '12px';
indexStatus.style.color = '#444';

function getActiveTabUrl(): Promise<string> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      resolve(tabs[0]?.url || "");
    });
  });
}

async function updateIndexStatus() {
  try {
    const url = await getActiveTabUrl();
    if (!url) {
      indexStatus.textContent = '';
      return;
    }
    const resp = await fetch(`${BACKEND_BASE_URL}/is-indexed?url=${encodeURIComponent(url)}`);
    const data = await resp.json();
    const indexed = !!data?.indexed;
    const host = data?.host || '';
    indexStatus.innerHTML = '';
    if (indexed) {
      const check = document.createElement('span');
      check.textContent = '✔';
      check.style.color = '#2e7d32';
      check.style.fontWeight = 'bold';
      const label = document.createElement('span');
      label.textContent = host ? `Indexed (${host})` : 'Indexed';
      label.style.color = '#2e7d32';
      indexStatus.appendChild(check);
      indexStatus.appendChild(label);
    } else {
      const btn = document.createElement('button');
      btn.id = 'btn-index-site';
      btn.textContent = 'Index this site';
      btn.style.background = '#fff';
      btn.style.color = '#222';
      btn.style.border = '1px solid #b0b0b0';
      btn.style.borderRadius = '6px';
      btn.style.padding = '4px 8px';
      btn.style.cursor = 'pointer';
      btn.onclick = () => {};
      indexStatus.appendChild(btn);
    }
  } catch (e) {
    // Silent fail; leave status empty
    indexStatus.textContent = '';
  }
}

// --- SmartQA log streaming ---
let smartqaLogSocket: WebSocket | null = null;
let streamingActive = false;
let currentAnswerBuffer = "";
let currentAnswerBubble: HTMLElement | null = null;
let pendingDeltaQueue = "";
let typewriterInterval: number | null = null;
// When false, ignore any late streaming events for the last question
let acceptStreaming = false;
// Incremented per ask; used to tag DOM nodes for cleanup
let currentSessionId = 0;

function renderLLMLinksMessage(llmMessage: string, links: {text: string, href: string}[], sessionId?: number) {
    const prev = document.getElementById('llm-links-block');
    if (prev) prev.remove();

    const chatDiv = document.getElementById("chat")!;
    const block = document.createElement('div');
    if (sessionId !== undefined) {
      block.setAttribute('data-session-id', String(sessionId));
    }
    block.id = 'llm-links-block';
    block.className = 'system-message';
    block.style.margin = '16px 0';
    const messageDiv = document.createElement('div');
    messageDiv.style.marginBottom = '6px';
    messageDiv.id = 'llm-links-message-span';
    messageDiv.textContent = llmMessage;
    const list = document.createElement('ul');
    list.style.marginTop = '4px';
    list.style.marginBottom = '4px';
    list.style.paddingLeft = '16px';
    list.innerHTML = links.map(link => `
      <li>
        <a href="${link.href}" target="_blank" rel="noopener noreferrer">${link.text || link.href}</a>
        <br>
        <small style="color:#888;">${link.href}</small>
      </li>
    `).join('');
    block.appendChild(messageDiv);
    block.appendChild(list);
    chatDiv.appendChild(block);
    block.scrollIntoView({behavior: "smooth"});
}



function ensureStreamingBubble(): HTMLElement {
  // Reuse in-memory bubble if it matches current session and is not a final bubble
  if (
    currentAnswerBubble &&
    currentAnswerBubble.getAttribute('data-session-id') === String(currentSessionId) &&
    !currentAnswerBubble.classList.contains('final')
  ) {
    return currentAnswerBubble;
  }
  // Otherwise, try to find an existing streaming bubble for this session in the DOM
  const existing = chatDiv.querySelector(
    `div.bubble.bot[data-session-id="${currentSessionId}"]:not(.final)`
  ) as HTMLElement | null;
  if (existing) {
    currentAnswerBubble = existing;
    return existing;
  }
  // Create an empty bot bubble to stream into
  const bubble = document.createElement('div');
  bubble.className = 'bubble bot';
  bubble.setAttribute('data-session-id', String(currentSessionId));
  bubble.classList.add('processing');
  const span = document.createElement('span');
  span.className = 'stream-text';
  bubble.appendChild(span);
  chatDiv.appendChild(bubble);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  currentAnswerBubble = bubble;
  return bubble;
}

function renderStreamedBufferAsMarkdown() {
  // Intentionally no-op to avoid replacing streamed chunks. Final answer rendered separately.
}

function connectSmartQALogSocket(logContainer: HTMLElement) {
  if (smartqaLogSocket) {
    smartqaLogSocket.close();
  }
  // Allow streaming for the newly initiated request
  acceptStreaming = true;
  smartqaLogSocket = new WebSocket("ws://localhost:5000/ws/smartqa-logs");
smartqaLogSocket.onmessage = (event) => {
    let isJSON = false;
    let msg: any;
    try {
        msg = JSON.parse(event.data);
        isJSON = true;
    } catch (e) {}

    // 1. LLM links (streaming-aware)
    if (isJSON && msg && msg.type === "llm_links_reset") {
        if (!acceptStreaming) return;
        // Create/reset the streaming block with provided links, empty message
        renderLLMLinksMessage("", (msg.links || []), currentSessionId);
        return;
    }
    if (isJSON && msg && msg.type === "llm_links_delta") {
        if (!acceptStreaming) return;
        const block = document.getElementById('llm-links-block');
        if (!block) return;
        let span = document.getElementById('llm-links-message-span');
        if (!span) {
          span = document.createElement('div');
          span.id = 'llm-links-message-span';
          block.insertBefore(span, block.firstChild);
        }
        span.textContent = (span.textContent || "") + (msg.text || "");
        chatDiv.scrollTop = chatDiv.scrollHeight;
        return;
    }
    if (isJSON && msg && msg.type === "llm_links_done") {
        if (!acceptStreaming) return;
        // Optionally mark final state
        const block = document.getElementById('llm-links-block');
        if (block) block.classList.add('final');
        return;
    }
    if (isJSON && msg && msg.type === "llm_links_message") {
        // Fallback non-streaming single-shot
        renderLLMLinksMessage(msg.message, msg.links, currentSessionId);
        return;
    }
    if (isJSON && msg && msg.type === "selected_links") {
        renderUsefulLinks(msg.links, currentSessionId);
        return;
    }

    // 2. Streamed answer events
    if (isJSON && msg && msg.type === "answer_reset") {
      if (!acceptStreaming) return;
      streamingActive = true;
      currentAnswerBuffer = "";
      pendingDeltaQueue = "";
      if (typewriterInterval !== null) {
        clearInterval(typewriterInterval);
        typewriterInterval = null;
      }
      // Ensure a streaming bubble exists but do NOT clear existing content.
      // This allows new data to be appended rather than replacing prior content.
      ensureStreamingBubble();
      return;
    }
    if (isJSON && msg && msg.type === "answer_delta") {
      if (!acceptStreaming) return;
      if (!streamingActive) return;
      const delta = msg.text || "";
      // Append immediately without throttling to avoid UI delay
      currentAnswerBuffer += delta;
      const bubble = ensureStreamingBubble();
      let span = bubble.querySelector('span.stream-text') as HTMLElement | null;
      if (!span) {
        span = document.createElement('span');
        span.className = 'stream-text';
        bubble.appendChild(span);
      }
      span.textContent += delta;
      chatDiv.scrollTop = chatDiv.scrollHeight;
      return;
    }
    if (isJSON && msg && msg.type === "answer_done") {
      if (!acceptStreaming) return;
      if (!streamingActive) return;
      streamingActive = false;
      // If queue is empty, finalize immediately. Otherwise the interval will finalize when drained.
      // Since we append immediately, there may be no pending queue or interval to drain
      if (typewriterInterval !== null) {
        clearInterval(typewriterInterval);
        typewriterInterval = null;
      }
      // Do not modify the streaming bubble on done; final HTTP answer will be a separate bubble
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
    acceptStreaming = false;
  };
}

// --- DROPDOWN to toggle between Ask and Ask Smart ---
// Remove dropdown and options
// const dropdown = document.createElement("select");
// dropdown.id = "ask-mode";
// dropdown.style.marginLeft = "8px";
// const optionAsk = document.createElement("option");
// optionAsk.value = "ask";
// optionAsk.textContent = "Ask";
// dropdown.appendChild(optionAsk);
// const optionSmart = document.createElement("option");
// optionSmart.value = "smart";
// optionSmart.textContent = "Ask Smart";
// dropdown.appendChild(optionSmart);
// const optionGemini = document.createElement("option");
// optionGemini.value = "gemini";
// optionGemini.textContent = "Ask Gemini";
// dropdown.appendChild(optionGemini);

// --- Mode selection buttons ---
const btnContainer = document.createElement('div');
btnContainer.style.display = 'inline-block';
btnContainer.style.marginLeft = '8px';

const btnSmart = document.createElement('button');
btnSmart.textContent = 'Search in this website';
btnSmart.id = 'btn-smart';
btnSmart.style.marginRight = '4px';

const btnWebsiteRag = document.createElement('button');
btnWebsiteRag.textContent = 'Website RAG';
btnWebsiteRag.id = 'btn-website-rag';
btnWebsiteRag.style.marginRight = '4px';

const btnGemini = document.createElement('button');
btnGemini.textContent = 'Google search';
btnGemini.id = 'btn-gemini';

btnContainer.appendChild(btnSmart);
btnContainer.appendChild(btnWebsiteRag);
btnContainer.appendChild(btnGemini);

// Selection state
let selectedMode: 'gemini' | 'smart' | 'website_rag' = 'gemini';

function updateButtonStyles() {
  // Reset styles
  [btnSmart, btnWebsiteRag, btnGemini].forEach(btn => {
    btn.style.background = '#fff';
    btn.style.color = '#222';
    btn.style.border = '1px solid #b0b0b0';
    btn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.04)';
    btn.style.borderRadius = '6px';
    btn.style.transition = 'background 0.15s, color 0.15s';
    btn.style.cursor = 'pointer';
    btn.onmouseover = null;
    btn.onmouseout = null;
  });
  // Selected styles
  if (selectedMode === 'smart') {
    btnSmart.style.background = '#1976d2';
    btnSmart.style.color = '#fff';
    btnSmart.style.border = '1.5px solid #1976d2';
    btnSmart.onmouseover = null;
    btnSmart.onmouseout = null;
    btnWebsiteRag.onmouseover = () => btnWebsiteRag.style.background = '#f3f3f3';
    btnWebsiteRag.onmouseout = () => btnWebsiteRag.style.background = '#fff';
    btnGemini.onmouseover = () => btnGemini.style.background = '#f3f3f3';
    btnGemini.onmouseout = () => btnGemini.style.background = '#fff';
  } else if (selectedMode === 'website_rag') {
    btnWebsiteRag.style.background = '#9c27b0';
    btnWebsiteRag.style.color = '#fff';
    btnWebsiteRag.style.border = '1.5px solid #9c27b0';
    btnWebsiteRag.onmouseover = null;
    btnWebsiteRag.onmouseout = null;
    btnSmart.onmouseover = () => btnSmart.style.background = '#f3f3f3';
    btnSmart.onmouseout = () => btnSmart.style.background = '#fff';
    btnGemini.onmouseover = () => btnGemini.style.background = '#f3f3f3';
    btnGemini.onmouseout = () => btnGemini.style.background = '#fff';
  } else {
    btnGemini.style.background = '#1976d2';
    btnGemini.style.color = '#fff';
    btnGemini.style.border = '1.5px solid #1976d2';
    btnGemini.onmouseover = null;
    btnGemini.onmouseout = null;
    btnSmart.onmouseover = () => btnSmart.style.background = '#f3f3f3';
    btnSmart.onmouseout = () => btnSmart.style.background = '#fff';
    btnWebsiteRag.onmouseover = () => btnWebsiteRag.style.background = '#f3f3f3';
    btnWebsiteRag.onmouseout = () => btnWebsiteRag.style.background = '#fff';
  }
}

btnSmart.onclick = () => {
  selectedMode = 'smart';
  updateButtonStyles();
};
btnWebsiteRag.onclick = () => {
  selectedMode = 'website_rag';
  updateButtonStyles();
};
btnGemini.onclick = () => {
  selectedMode = 'gemini';
  updateButtonStyles();
};

// Set default selection
updateButtonStyles();

// --- Smart Ask button (always visible, no dropdown) ---
const smartBtn = document.createElement("button");
smartBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="#fff"/></svg>`;
smartBtn.title = "Send";
smartBtn.id = "smart-ask-btn";
smartBtn.style.marginLeft = "8px";
smartBtn.style.display = "inline-block";

const inputRow = document.getElementById("inputRow");
if (inputRow) {
  inputRow.appendChild(btnContainer);
  inputRow.appendChild(smartBtn);
  inputRow.appendChild(indexStatus);
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
function appendMessage(text: string, sender: 'user' | 'bot' | 'thinking', sessionId?: number): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  if (sessionId !== undefined) {
    bubble.setAttribute('data-session-id', String(sessionId));
  }

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
        // Only set innerHTML for non-streaming appended messages (final answers, info)
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



// --- Smart Ask button logic ---
smartBtn.onclick = async function () {
  // Start a new session for this question
  currentSessionId += 1;
  
  const question = questionInput.value.trim();
  if (!question) return;

  // Get selected tool
  let selectedTool = selectedMode;
  if (!selectedTool) selectedTool = 'gemini'; // fallback
  console.log('Selected tool:', selectedTool);

  // Add mode tag above the bubble
  const tag = document.createElement('div');
  tag.textContent = selectedTool === 'gemini' ? "Gemini QA" : (selectedTool === 'smart' ? "Smart QA" : "Website RAG");
  tag.style.fontSize = "0.8em";
  tag.style.color = selectedTool === 'gemini' ? "#0af" : (selectedTool === 'smart' ? "#0a5" : "#9c27b0");
  tag.style.fontWeight = "bold";
  tag.style.letterSpacing = "0.03em";
  tag.style.margin = "0 10px 0 0";
  tag.style.textAlign = "right";
  tag.style.width = "100%";
  chatDiv.appendChild(tag);
  tag.setAttribute('data-session-id', String(currentSessionId));
  appendMessage(question, "user", currentSessionId);
  const thinkingBubble = appendMessage("Thinking...", "thinking", currentSessionId);
  const logContainer = (thinkingBubble as any).logContainer;
  logContainer.innerText = "";
  if (selectedTool === 'smart') {
    connectSmartQALogSocket(logContainer);
  } // No log socket for Gemini or Site Memory placeholder

  // --- NEW: Always use getPageDataFromActiveTab to ensure injection ---
  const pageData = await getPageDataFromActiveTab();
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const tab = tabs[0];
    const page_url = tab?.url ?? "";
    let domain = "";
    try {
      domain = page_url ? new URL(page_url).hostname : "";
    } catch {}
    const body: any = {
      text: pageData.text,
      question,
      links: pageData.links,
      page_url,
    };
    if (selectedTool === 'website_rag') {
      body.domain = domain;
    }
    try {
      let endpoint = "http://localhost:5000/ask-smart";
      if (selectedTool === 'gemini') {
        endpoint = "http://localhost:5000/ask-gemini";
      } else if (selectedTool === 'website_rag') {
        endpoint = "http://localhost:5000/ask-website-rag";
      }
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      (thinkingBubble as any).thinkingText.textContent = 'Completed thinking.';
      if (smartqaLogSocket && selectedTool === 'smart') {
        // Stop accepting any more streamed content for this request
        acceptStreaming = false;
        smartqaLogSocket.close();
      }

      // Show the answer
      // Smart and Website RAG paths: append a distinct final answer bubble in light green
      if (selectedTool === 'smart' || selectedTool === 'website_rag') {
        if (data.answer) {
          const finalBubble = appendMessage(data.answer, 'bot', currentSessionId);
          finalBubble.classList.add('final');
        }
      } else {
        // Gemini path (non-streaming): render HTTP answer directly
        appendMessage(data.answer, 'bot', currentSessionId);
      }
      // Show sources (if any)
      if (data.sources && data.sources.length > 0) {
        // Tag sources with the session id by wrapping in a container
        const sourcesWrapper = document.createElement('div');
        sourcesWrapper.setAttribute('data-session-id', String(currentSessionId));
        chatDiv.appendChild(sourcesWrapper);
        renderSources(data.sources);
      }

      // If not sufficient, show LLM-picked links (plain, no style)
      if (data.sufficient === false && data.selected_links && data.selected_links.length > 0) {
        appendMessage("Try checking one of these links for more info:", "bot", currentSessionId);
        data.selected_links.forEach((l: any) => {
          appendMessage(`• ${l.text} — ${l.href}`, "bot", currentSessionId);
        });
      }

      if (data.visited_urls && data.visited_urls.length > 0) {
        const urls = data.visited_urls;
        let msg = "Pages visited:\n";
        urls.forEach((url: string, idx: number) => {
          msg += `• ${url}\n`;
        });
        msg += `\nYou can also view the last visited page for more details..`;
        appendMessage(msg, "bot", currentSessionId);
      }

    } catch (err) {
      (thinkingBubble as any).thinkingText.textContent = selectedTool === 'gemini' ? "Error (Gemini QA). Please try again." : "Error (smart QA). Please try again.";
      acceptStreaming = false;
    }
    questionInput.value = "";
    // Reset streaming state for next question
    streamingActive = false;
    currentAnswerBuffer = "";
    currentAnswerBubble = null;
  });
};

// Keep index status updated
updateIndexStatus();
chrome.tabs.onActivated.addListener(() => {
  updateIndexStatus();
});
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (tab.active && (changeInfo.status === 'complete' || changeInfo.url)) {
    updateIndexStatus();
  }
});
