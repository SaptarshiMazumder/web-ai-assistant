import { marked } from "marked";
import hljs from "highlight.js";
import { crawlEntireSite, debugLog } from "./siteCrawler";


// crawl webpage start

const BACKEND_BASE_URL = "http://localhost:5000";
let indexPollTimer: number | null = null;
let indexingInProgress = false;
let lastIndexUrl: string | null = null;

const chatTab = document.getElementById("tab-chat") as HTMLButtonElement | null;
const conversationsTab = document.getElementById("tab-conversations") as HTMLButtonElement | null;
const chatPanel = document.getElementById("chat-panel") as HTMLElement | null;
const conversationsPanel = document.getElementById("conversations-panel") as HTMLElement | null;
const botKeyInput = document.getElementById("botKeyInput") as HTMLInputElement | null;
const botKeySave = document.getElementById("botKeySave") as HTMLButtonElement | null;
const conversationList = document.getElementById("conversation-list") as HTMLElement | null;
const conversationDetail = document.getElementById("conversation-detail") as HTMLElement | null;
const convPrevBtn = document.getElementById("conv-prev") as HTMLButtonElement | null;
const convNextBtn = document.getElementById("conv-next") as HTMLButtonElement | null;
const CONV_KEY = "webai_conversations_bot_key";
let convCursorStack: string[] = [];
let convNextCursor: string | null = null;
let convCurrentCursor: string | null = null;
const convStatusById: Record<string, string> = {};

function setActiveTab(tab: "chat" | "conversations") {
  if (chatTab && conversationsTab) {
    chatTab.classList.toggle("active", tab === "chat");
    conversationsTab.classList.toggle("active", tab === "conversations");
  }
  if (chatPanel && conversationsPanel) {
    chatPanel.classList.toggle("hidden", tab !== "chat");
    conversationsPanel.classList.toggle("hidden", tab !== "conversations");
  }
  if (tab === "conversations") {
    fetchConversationList();
  }
}

function readBotKey(): string {
  if (botKeyInput && botKeyInput.value.trim()) return botKeyInput.value.trim();
  try {
    return localStorage.getItem(CONV_KEY) || "";
  } catch {
    return "";
  }
}

function saveBotKey(key: string) {
  try {
    localStorage.setItem(CONV_KEY, key);
  } catch {
    // ignore
  }
}

function formatTime(ts: string | null | undefined): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

function renderConversationList(sessions: any[]) {
  if (!conversationList) return;
  conversationList.innerHTML = "";
  if (!sessions || sessions.length === 0) {
    const empty = document.createElement("div");
    empty.textContent = "No conversations yet.";
    empty.style.color = "#64748b";
    empty.style.padding = "8px";
    conversationList.appendChild(empty);
    return;
  }
  sessions.forEach((s: any) => {
    if (s && s.session_id) convStatusById[s.session_id] = s.status || "";
    const item = document.createElement("div");
    item.className = "conv-item";
    const title = document.createElement("div");
    title.className = "conv-title";
    title.textContent = s.title || s.site_title || s.site_url || s.session_id;
    const meta = document.createElement("div");
    meta.className = "conv-meta";
    let statusLabel = "";
    if (s.status && s.status !== "active") {
      statusLabel = "Session ended";
    } else {
      const last = new Date(s.last_active_at || "");
      if (!Number.isNaN(last.getTime())) {
        const diffMin = Math.floor((Date.now() - last.getTime()) / 60000);
        if (diffMin <= 5) statusLabel = "Active";
        else if (diffMin <= 30) statusLabel = "Inactive";
      }
    }
    meta.textContent = `${statusLabel ? statusLabel + " • " : ""}${s.message_count || 0} msgs • ${formatTime(s.last_active_at)}`;
    item.appendChild(title);
    item.appendChild(meta);
    item.addEventListener("click", () => fetchConversationDetail(s.session_id));
    conversationList.appendChild(item);
  });
}

function renderConversationDetail(messages: any[], statusLabel: string) {
  if (!conversationDetail) return;
  conversationDetail.innerHTML = "";
  const back = document.createElement("button");
  back.className = "conv-back";
  back.textContent = "Back to list";
  back.onclick = () => {
    conversationDetail.classList.add("hidden");
    if (conversationList) conversationList.classList.remove("hidden");
  };
  conversationDetail.appendChild(back);
  const count = document.createElement("div");
  count.className = "conv-count";
  count.textContent = `Messages: ${messages?.length || 0}`;
  conversationDetail.appendChild(count);

  const status = document.createElement("div");
  status.className = "conv-count";
  status.textContent = `Status: ${statusLabel}`;
  conversationDetail.appendChild(status);

  function formatMessageTime(ts?: string | null) {
    const d = ts ? new Date(ts) : new Date();
    if (Number.isNaN(d.getTime())) return "";
    const now = new Date();
    const isToday = d.toDateString() === now.toDateString();
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const isYesterday = d.toDateString() === yesterday.toDateString();
    const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (isToday) return `Today ${time}`;
    if (isYesterday) return `Yesterday ${time}`;
    return `${d.toLocaleDateString()} ${time}`;
  }

  function dateKey(ts?: string | null) {
    if (!ts) return "";
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return "";
    return d.toDateString();
  }

  const list = messages || [];
  list.forEach((m: any, idx: number) => {
    const prev = list[idx - 1];
    const showDate = dateKey(m.created_at) !== dateKey(prev?.created_at);
    if (showDate) {
      const sep = document.createElement("div");
      sep.className = "conv-date";
      sep.textContent = dateKey(m.created_at) || new Date().toDateString();
      conversationDetail.appendChild(sep);
    }
    const row = document.createElement("div");
    row.className = `conv-message ${m.role === "user" ? "user" : "bot"}`;
    row.textContent = m.content || "";
    const meta = document.createElement("div");
    meta.className = "conv-time";
    meta.textContent = formatMessageTime(m.created_at || null);
    row.appendChild(meta);
    conversationDetail.appendChild(row);
  });
}

function updatePaginationControls() {
  if (convPrevBtn) convPrevBtn.disabled = convCursorStack.length === 0;
  if (convNextBtn) convNextBtn.disabled = !convNextCursor;
}

async function fetchConversationList(cursor?: string | null) {
  if (!conversationList) return;
  const pk = readBotKey();
  if (!pk) {
  const statusLabel = convStatusById[sessionId] || "Unknown";
    conversationList.innerHTML = "<div style='color:#64748b;padding:8px;'>Enter a bot publishable key to view conversations.</div>";
    convNextCursor = null;
    convCursorStack = [];
    convCurrentCursor = null;
    updatePaginationControls();
    return;
  }
  if (botKeyInput) botKeyInput.value = pk;
  try {
    const cursorParam = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
    const resp = await fetch(`${BACKEND_BASE_URL}/v1/pk/${encodeURIComponent(pk)}/conversations?limit=50${cursorParam}`);
    if (!resp.ok) {
      conversationList.innerHTML = `<div style='color:#b91c1c;padding:8px;'>Failed to load conversations.</div>`;
      convNextCursor = null;
      updatePaginationControls();
      return;
    }
    const data = await resp.json();
    renderConversationList(data.sessions || []);
    convNextCursor = data.next_cursor || null;
    convCurrentCursor = cursor || null;
    updatePaginationControls();
  } catch (e) {
    conversationList.innerHTML = `<div style='color:#b91c1c;padding:8px;'>Error loading conversations.</div>`;
    convNextCursor = null;
    updatePaginationControls();
  }
}

async function fetchConversationDetail(sessionId: string) {
  if (!conversationDetail || !conversationList) return;
  const pk = readBotKey();
  if (!pk) return;
  try {
    const resp = await fetch(`${BACKEND_BASE_URL}/v1/pk/${encodeURIComponent(pk)}/conversations/${encodeURIComponent(sessionId)}?limit=200`);
    if (!resp.ok) return;
    const data = await resp.json();
    conversationList.classList.add("hidden");
    conversationDetail.classList.remove("hidden");
    renderConversationDetail(data.messages || [], statusLabel);
    const endBtn = document.createElement("button");
    endBtn.className = "conv-back";
    endBtn.textContent = "End session";
    endBtn.onclick = async () => {
      try {
        await fetch(`${BACKEND_BASE_URL}/v1/pk/${encodeURIComponent(pk)}/conversations/${encodeURIComponent(sessionId)}/end`, {
          method: "POST",
        });
      } finally {
        conversationDetail.classList.add("hidden");
        if (conversationList) conversationList.classList.remove("hidden");
        fetchConversationList(convCurrentCursor);
      }
    };
    conversationDetail.insertBefore(endBtn, conversationDetail.firstChild);
  } catch {
    // ignore
  }
}

if (chatTab) chatTab.addEventListener("click", () => setActiveTab("chat"));
if (conversationsTab) conversationsTab.addEventListener("click", () => setActiveTab("conversations"));
if (botKeySave && botKeyInput) {
  botKeySave.addEventListener("click", () => {
    const key = botKeyInput.value.trim();
    if (!key) return;
    saveBotKey(key);
    convCursorStack = [];
    fetchConversationList(null);
  });
}

if (convNextBtn) {
  convNextBtn.addEventListener("click", () => {
    if (!convNextCursor) return;
    if (convCurrentCursor) convCursorStack.push(convCurrentCursor);
    fetchConversationList(convNextCursor);
  });
}

if (convPrevBtn) {
  convPrevBtn.addEventListener("click", () => {
    const prev = convCursorStack.pop() || null;
    fetchConversationList(prev);
  });
}

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

function renderUsefulLinks(links: { text: string, href: string }[], sessionId?: number) {
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
  block.scrollIntoView({ behavior: "smooth" });
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
    lastIndexUrl = url;

    // If an index job is running, show best-effort progress.
    if (indexingInProgress) {
      try {
        const st = await fetch(`${BACKEND_BASE_URL}/index-job-status?url=${encodeURIComponent(url)}`);
        const sd = await st.json();
        if (sd && sd.status === "ok") {
          const stage = sd.stage || "crawling";
          const count = typeof sd.pages_crawled === "number" ? sd.pages_crawled : 0;
          const depth = typeof sd.last_depth === "number" ? sd.last_depth : -1;
          // Keep the stop button, but add a small status label beside it.
          // We reuse indexStatus container; button is re-rendered below.
          indexStatus.setAttribute("data-progress", `${stage}|${count}|${depth}`);
        }
      } catch {
        // ignore
      }
    }
    const resp = await fetch(`${BACKEND_BASE_URL}/is-indexed?url=${encodeURIComponent(url)}`);
    const data = await resp.json();
    const indexed = !!data?.indexed;
    const host = data?.host || '';
    indexStatus.innerHTML = '';
    if (indexed) {
      indexingInProgress = false;
      if (indexPollTimer !== null) {
        clearInterval(indexPollTimer);
        indexPollTimer = null;
      }
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
      btn.textContent = indexingInProgress ? 'Stop crawling' : 'Index this site';
      btn.style.background = '#fff';
      btn.style.color = '#222';
      btn.style.border = '1px solid #b0b0b0';
      btn.style.borderRadius = '6px';
      btn.style.padding = '4px 8px';
      btn.style.cursor = 'pointer';
      btn.onclick = async () => {
        try {
          if (indexingInProgress) {
            // Request a graceful stop; backend will upload whatever has been crawled so far.
            btn.disabled = true;
            btn.textContent = 'Stopping crawl...';
            await fetch(`${BACKEND_BASE_URL}/cancel-index-site`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ url }),
            });
            indexingInProgress = false;
            btn.disabled = false;
            btn.textContent = 'Index this site';
            return;
          }

          indexingInProgress = true;
          btn.disabled = true;
          btn.textContent = 'Crawling...';
          await fetch(`${BACKEND_BASE_URL}/index-site`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
          });

          // Switch to "Stop indexing" state immediately
          btn.disabled = false;
          btn.textContent = 'Stop crawling';

          // Start polling until indexed
          if (indexPollTimer !== null) {
            clearInterval(indexPollTimer);
            indexPollTimer = null;
          }
          indexPollTimer = window.setInterval(async () => {
            try {
              const r = await fetch(`${BACKEND_BASE_URL}/is-indexed?url=${encodeURIComponent(url)}`);
              const d = await r.json();
              if (d && d.indexed) {
                indexingInProgress = false;
                clearInterval(indexPollTimer!);
                indexPollTimer = null;
                updateIndexStatus();
              }
            } catch { }
          }, 5000);
        } catch (e) {
          indexingInProgress = false;
          btn.disabled = false;
          btn.textContent = 'Index this site';
          console.error('Failed to start indexing', e);
        }
      };
      indexStatus.appendChild(btn);

      // Optional progress text (only while job is running)
      if (indexingInProgress) {
        const prog = document.createElement("span");
        prog.style.marginLeft = "6px";
        prog.style.fontSize = "12px";
        prog.style.color = "#666";
        const packed = indexStatus.getAttribute("data-progress") || "";
        const [stage, countStr, depthStr] = packed.split("|");
        const count = countStr ? Number(countStr) : 0;
        const depth = depthStr ? Number(depthStr) : -1;
        const stageLabel =
          stage === "uploading" ? "Uploading…" :
            stage === "importing" ? "Sending to Vertex RAG…" :
              stage === "import_submitted" ? "Vertex indexing…" :
                stage ? `${stage}…` : "Crawling…";
        prog.textContent = depth >= 0 ? `${stageLabel} (pages: ${count}, depth: ${depth})` : `${stageLabel} (pages: ${count})`;
        indexStatus.appendChild(prog);
      }
    }
  } catch (e) {
    // Silent fail; leave status empty
    indexStatus.textContent = '';
  }
}
let finalStreamId: string | null = null;
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

const streamBoxes: Record<string, HTMLElement> = {};
function getOrCreateStreamBox(parent: HTMLElement, streamId: string): HTMLElement {
  const id = `answer-stream-${streamId}`;
  let box = parent.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
  if (!box) {
    box = document.createElement('div');
    box.id = id;
    box.classList.add('streamed-raw-data');
    box.style.borderRadius = '8px';
    box.style.padding = '8px';
    box.style.margin = '8px 0';

    parent.appendChild(box);
    streamBoxes[streamId] = box;
  }
  return box;
}


function renderLLMLinksMessage(llmMessage: string, links: { text: string, href: string }[], sessionId?: number) {
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
  block.scrollIntoView({ behavior: "smooth" });
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

  // Per-connection state
  const streamBoxes: Record<string, HTMLElement> = {};
  const streamBuffers: Record<string, string> = {};
  const streamActive: Record<string, boolean> = {};
  let finalStreamId: string | null = null;

  function getOrCreateStreamBox(parent: HTMLElement, streamId: string): HTMLElement {
    const id = `answer-stream-${streamId}`;
    let box = parent.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
    if (!box) {
      box = document.createElement('div');
      box.id = id;
      box.classList.add('streamed-raw-data');
      box.style.borderRadius = '8px';
      box.style.padding = '8px';
      box.style.margin = '8px 0';
      parent.appendChild(box);
      streamBoxes[streamId] = box;
    }
    return box;
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
    } catch (e) { }

    // 1) LLM links (streaming-aware) — unchanged
    if (isJSON && msg && msg.type === "llm_links_reset") {
      if (!acceptStreaming) return;
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
      const block = document.getElementById('llm-links-block');
      if (block) block.classList.add('final');
      return;
    }
    if (isJSON && msg && msg.type === "llm_links_message") {
      renderLLMLinksMessage(msg.message, msg.links, currentSessionId);
      return;
    }
    if (isJSON && msg && msg.type === "selected_links") {
      renderUsefulLinks(msg.links, currentSessionId);
      return;
    }

    // 2) Streamed answer events (group by stream_id)
    if (isJSON && msg && msg.type === "answer_reset") {
      if (!acceptStreaming) return;
      if (finalStreamId) return; // already have a winner; ignore new streams

      // Reset global throttling (kept from your code)
      streamingActive = true;
      currentAnswerBuffer = "";
      pendingDeltaQueue = "";
      if (typewriterInterval !== null) {
        clearInterval(typewriterInterval);
        typewriterInterval = null;
      }

      // Big gray bubble (parent) stays the same
      const parentBubble = ensureStreamingBubble();
      if (!parentBubble) return;

      // Ensure per-stream child box
      const streamId: string = msg.stream_id || `default-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const box = getOrCreateStreamBox(parentBubble, streamId);
      box.innerHTML = ""; // reset only this stream's box
      streamBuffers[streamId] = "";
      streamActive[streamId] = true;
      return;
    }

    if (isJSON && msg && msg.type === "answer_delta") {
      if (!acceptStreaming) return;

      const parentBubble = ensureStreamingBubble();
      if (!parentBubble) return;

      const streamId: string = msg.stream_id || `default-${Date.now()}-${Math.random().toString(36).slice(2)}`;

      // If we already locked a final, ignore all others
      if (finalStreamId && streamId !== finalStreamId) return;

      // Only accept deltas for active streams
      if (!streamActive[streamId]) return;

      const box = getOrCreateStreamBox(parentBubble, streamId);
      const delta = msg.text || "";
      // Accumulate and render markdown. If any footer marker appears, stop accepting further deltas for this stream.
      if (/(^|\n)\s*(SUFFICIENT|CONFIDENCE|FULL\s*INFO)\b/i.test(streamBuffers[streamId] + delta)) {
        // Lock this stream and avoid appending footer markers
        streamActive[streamId] = false;
        // Truncate any existing buffer before the first footer marker
        const combined = streamBuffers[streamId] + delta;
        const cut = combined.search(/(^|\n)\s*(SUFFICIENT|CONFIDENCE|FULL\s*INFO)\b/i);
        streamBuffers[streamId] = cut !== -1 ? combined.slice(0, cut) : streamBuffers[streamId];
        // Render the truncated buffer once
        const safeRendered = streamBuffers[streamId]
          .replace(/(^|\n)\s*sufficient(\s*:.*)?(?=$|\n)/gim, "")
          .replace(/(^|\n)\s*confidence(\s*:.*)?(?=$|\n)/gim, "")
          .replace(/(^|\n)\s*full\s*info(\s*:.*)?(?=$|\n)/gim, "");
        const parsedNow = marked.parse(safeRendered);
        if (parsedNow instanceof Promise) {
          parsedNow.then(html => {
            box.innerHTML = html;
            box.querySelectorAll("pre code").forEach((block) => {
              hljs.highlightElement(block as HTMLElement);
            });
            chatDiv.scrollTop = chatDiv.scrollHeight;
          });
        } else {
          box.innerHTML = parsedNow;
          box.querySelectorAll("pre code").forEach((block) => {
            hljs.highlightElement(block as HTMLElement);
          });
          chatDiv.scrollTop = chatDiv.scrollHeight;
        }
        return;
      }

      // Strip footer/meta lines from this chunk anyway
      const sanitizedDelta = String(delta)
        // remove SUFFICIENT/CONFIDENCE/Full info lines
        .replace(/(^|\n)\s*sufficient(\s*:.*)?(?=$|\n)/gim, "")
        .replace(/(^|\n)\s*confidence(\s*:.*)?(?=$|\n)/gim, "")
        .replace(/(^|\n)\s*full\s*info(\s*:.*)?(?=$|\n)/gim, "")
        // remove placeholder or generic visit lines
        .replace(/^\s*please\s+visit\s*(\[?page url\]?|page url|\[page url\])\s*.*?$/gim, "")
        .replace(/^\s*please\s+visit\s*\[?page url\]?\s*for\s+the\s+full\s+details\.?\s*$/gim, "");
      streamBuffers[streamId] = (streamBuffers[streamId] || "") + sanitizedDelta;
      // Sanitize entire accumulated buffer to ensure footer lines never appear mid-stream
      // Ensure final rendering also excludes any footer/meta lines
      const fullySanitized = streamBuffers[streamId]
        .replace(/(^|\n)\s*sufficient(\s*:.*)?(?=$|\n)/gim, "")
        .replace(/(^|\n)\s*confidence(\s*:.*)?(?=$|\n)/gim, "")
        .replace(/(^|\n)\s*full\s*info(\s*:.*)?(?=$|\n)/gim, "")
        .replace(/(^|\n)\s*please\s+visit\s*(\[?page url\]?|page url|\[page url\]).*$/gim, "");
      const parsed = marked.parse(fullySanitized);
      if (parsed instanceof Promise) {
        parsed.then(html => {
          box.innerHTML = html;
          box.querySelectorAll("pre code").forEach((block) => {
            hljs.highlightElement(block as HTMLElement);
          });
          chatDiv.scrollTop = chatDiv.scrollHeight;
        });
      } else {
        box.innerHTML = parsed;
        box.querySelectorAll("pre code").forEach((block) => {
          hljs.highlightElement(block as HTMLElement);
        });
      }
      chatDiv.scrollTop = chatDiv.scrollHeight;
      return;
    }

    if (isJSON && msg && msg.type === "answer_done") {
      if (!acceptStreaming) return;

      const streamId: string = msg.stream_id || `default-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      streamActive[streamId] = false;

      // Keep global semantics for your existing timers
      if (streamingActive) {
        streamingActive = false;
        if (typewriterInterval !== null) {
          clearInterval(typewriterInterval);
          typewriterInterval = null;
        }
      }

      // If this stream is sufficient and we haven't chosen a final yet:
      if (msg.sufficient === true && !finalStreamId) {
        finalStreamId = streamId;

        const parentBubble = ensureStreamingBubble();
        if (parentBubble) {
          // Remove non-winning child boxes directly under the bubble
          Array.from(parentBubble.children).forEach((child) => {
            const keep = (child as HTMLElement).id === `answer-stream-${finalStreamId}`;
            if (!keep) parentBubble.removeChild(child);
          });

          const winner = streamBoxes[finalStreamId];
          if (winner) {
            winner.style.boxShadow = '';
            winner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
      }
      // If not sufficient, do nothing; its box remains until a winner arrives (then it's purged)
      return;
    }

    // 3) All other logs: append as a <div> (NOT with innerText)
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
    } catch { }
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
