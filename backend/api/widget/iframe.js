(() => {
  const params = new URLSearchParams(location.search);
  const pk = params.get("pk") || "";
  const apiBase = params.get("apiBase") || location.origin;
  const siteUrl = params.get("siteUrl") || document.referrer || "";
  const siteTitle = params.get("siteTitle") || "";
  const color = params.get("color") || "#1976d2";
  const title = params.get("title") || "Chat";
  const textColor = params.get("textColor") || "#ffffff";
  const placeholder = params.get("placeholder") || "Ask a question...";
  const theme = (params.get("theme") || "light").toLowerCase();
  const headerIconUrl = params.get("headerIcon") || "";
  const welcomeMessage = params.get("welcomeMessage") || params.get("welcome_message") || "Welcome! How can I help you today?";
  const footerMessage = params.get("footer") || params.get("footerMessage") || "Powered by WebAI";
  const displaySources = parseBool(params.get("displaySources"), false);
  const sourcesLabel = params.get("sourcesLabel") || "Sources";

  function parseBool(val, def) {
    if (val == null || val === "") return def;
    const v = String(val).toLowerCase();
    return v === "true" || v === "yes" || v === "1";
  }

  const SOFT_WRAP_TOKEN_MIN = 32;
  const SOFT_WRAP_CHUNK = 24;
  const SOFT_WRAP_SEPARATORS = /[\/\-\_\.\?\&\=\#\:\@]/;

  function appendSoftWrappedText(target, text) {
    if (text == null) return;
    const parts = String(text).split(/(\s+)/);
    parts.forEach((part) => {
      if (!part) return;
      if (/^\s+$/.test(part)) {
        target.appendChild(document.createTextNode(part));
        return;
      }
      if (part.length <= SOFT_WRAP_TOKEN_MIN) {
        target.appendChild(document.createTextNode(part));
        return;
      }
      const pieces = part.split(/([\/\-\_\.\?\&\=\#\:\@])/);
      let run = 0;
      pieces.forEach((piece) => {
        if (!piece) return;
        const isSep = piece.length === 1 && SOFT_WRAP_SEPARATORS.test(piece);
        target.appendChild(document.createTextNode(piece));
        run += piece.length;
        if (isSep || run >= SOFT_WRAP_CHUNK) {
          target.appendChild(document.createElement("wbr"));
          run = 0;
        }
      });
    });
  }

  const DEFAULT_SUGGESTIONS = ["What can you do?", "Ask a question", "Get help"];
  const WELCOME_TEXT = welcomeMessage;

  const chat = document.getElementById("chat");
  const messagesEl = document.getElementById("messages");
  const welcomeEl = document.getElementById("welcome");
  const input = document.getElementById("msg");
  const send = document.getElementById("send");
  const footerEl = document.getElementById("footer");
  const headerIconSlot = document.getElementById("headerIcon");
  const headerTitleEl = document.getElementById("headerTitle");
  const wrap = document.querySelector(".wrap");
  const row = document.querySelector(".row");

  document.documentElement.style.setProperty("--widget-color", color);
  document.documentElement.style.setProperty("--widget-text-color", textColor);

  if (headerTitleEl) headerTitleEl.textContent = title;
  if (headerIconSlot) {
    if (headerIconUrl && !headerIconUrl.startsWith("blob:")) {
      const img = document.createElement("img");
      img.src = headerIconUrl;
      img.alt = "";
      img.className = "header-icon";
      headerIconSlot.appendChild(img);
    } else {
      const div = document.createElement("div");
      div.className = "header-icon-default";
      div.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>';
      headerIconSlot.appendChild(div);
    }
  }
  if (input) input.placeholder = placeholder;
  if (footerEl) footerEl.textContent = footerMessage;
  if (theme === "dark") {
    document.body.setAttribute("data-theme", "dark");
    if (wrap) wrap.style.background = "#1e293b";
    if (chat) chat.style.background = "#1e293b";
    if (row) row.style.background = "#0f172a";
    if (row) row.style.borderColor = "#334155";
  }
  document.title = title;

  function renderWelcome() {
    if (!welcomeEl) return;
    welcomeEl.innerHTML = "";
    const rowDiv = document.createElement("div");
    rowDiv.className = "message-row";
    if (headerIconUrl && !headerIconUrl.startsWith("blob:")) {
      const av = document.createElement("img");
      av.src = headerIconUrl;
      av.alt = "";
      av.className = "message-avatar";
      rowDiv.appendChild(av);
    } else {
      const av = document.createElement("div");
      av.className = "message-avatar-default";
      av.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>';
      rowDiv.appendChild(av);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble bot";
    appendSoftWrappedText(bubble, WELCOME_TEXT);
    rowDiv.appendChild(bubble);
    welcomeEl.appendChild(rowDiv);

    const suggestionsDiv = document.createElement("div");
    suggestionsDiv.className = "suggestions";
    DEFAULT_SUGGESTIONS.forEach((q) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = q;
      btn.addEventListener("click", () => {
        input.value = q;
        sendMessage();
      });
      suggestionsDiv.appendChild(btn);
    });
    welcomeEl.appendChild(suggestionsDiv);
  }

  function hideWelcome() {
    if (welcomeEl) welcomeEl.style.display = "none";
  }

  function appendTypingBubble() {
    removeTypingBubble();
    const rowDiv = document.createElement("div");
    rowDiv.className = "message-row";
    rowDiv.id = "typing-row";
    rowDiv.appendChild(botAvatarEl());
    const bubble = document.createElement("div");
    bubble.className = "bubble bot typing-bubble";
    bubble.innerHTML = "<span></span><span></span><span></span>";
    rowDiv.appendChild(bubble);
    if (messagesEl) messagesEl.appendChild(rowDiv);
    if (chat) chat.scrollTop = chat.scrollHeight;
  }

  function removeTypingBubble() {
    const el = document.getElementById("typing-row");
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  renderWelcome();

  function botAvatarEl() {
    if (headerIconUrl && !headerIconUrl.startsWith("blob:")) {
      const av = document.createElement("img");
      av.src = headerIconUrl;
      av.alt = "";
      av.className = "message-avatar";
      return av;
    }
    const av = document.createElement("div");
    av.className = "message-avatar-default";
    av.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>';
    return av;
  }

  function appendBubble(text, who, citations) {
    hideWelcome();

    const rowDiv = document.createElement("div");
    rowDiv.className = "message-row " + (who === "user" ? "user-row" : "");

    if (who === "bot") {
      rowDiv.appendChild(botAvatarEl());
    }

    const div = document.createElement("div");
    div.className = "bubble " + who;
    setBubbleText(div, text);
    if (who === "user") {
      div.style.background = color;
      div.style.color = textColor;
    }
    if (citations && citations.length && displaySources) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.innerHTML =
        "<div><b>" + (sourcesLabel.replace(/</g, "&lt;").replace(/>/g, "&gt;")) + "</b></div>" +
        citations
          .slice(0, 6)
          .map((c) => {
            const url = (c && c.url) || "";
            if (!url) return "";
            const safe = url.replace(/"/g, "&quot;");
            return '<div><a href="' + safe + '" target="_blank" rel="noopener noreferrer">' + safe + "</a></div>";
          })
          .join("");
      div.appendChild(meta);
    }
    rowDiv.appendChild(div);

    if (messagesEl) messagesEl.appendChild(rowDiv);
    if (chat) chat.scrollTop = chat.scrollHeight;
  }

  function setBubbleText(bubble, text) {
    bubble.innerHTML = "";
    appendSoftWrappedText(bubble, text);
  }

  function ensureStreamingBubble() {
    hideWelcome();
    const rowDiv = document.createElement("div");
    rowDiv.className = "message-row";
    rowDiv.appendChild(botAvatarEl());
    const bubble = document.createElement("div");
    bubble.className = "bubble bot";
    rowDiv.appendChild(bubble);
    if (messagesEl) messagesEl.appendChild(rowDiv);
    if (chat) chat.scrollTop = chat.scrollHeight;
    return bubble;
  }

  function appendCitationsToBubble(bubble, citations) {
    if (!citations || !citations.length || !displaySources) return;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      "<div><b>" + (sourcesLabel.replace(/</g, "&lt;").replace(/>/g, "&gt;")) + "</b></div>" +
      citations
        .slice(0, 6)
        .map((c) => {
          const url = (c && c.url) || "";
          if (!url) return "";
          const safe = url.replace(/"/g, "&quot;");
          return '<div><a href="' + safe + '" target="_blank" rel="noopener noreferrer">' + safe + "</a></div>";
        })
        .join("");
    bubble.appendChild(meta);
  }

  const STREAM_TICK_MS = 24;
  const STREAM_CHARS_PER_TICK = 3;

  async function streamResponse(resp) {
    if (!resp.body) throw new Error("No response body");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let pending = "";
    let ticking = false;
    let doneEvent = null;
    const bubble = ensureStreamingBubble();
    function startTicker() {
      if (ticking) return;
      ticking = true;
      const tick = () => {
        if (pending.length > 0) {
          const slice = pending.slice(0, STREAM_CHARS_PER_TICK);
          pending = pending.slice(STREAM_CHARS_PER_TICK);
          text += slice;
          setBubbleText(bubble, text);
          if (chat) chat.scrollTop = chat.scrollHeight;
          setTimeout(tick, STREAM_TICK_MS);
          return;
        }
        ticking = false;
        if (doneEvent) {
          text = doneEvent.answer || text;
          setBubbleText(bubble, text);
          appendCitationsToBubble(bubble, doneEvent.citations || []);
          if (chat) chat.scrollTop = chat.scrollHeight;
          doneEvent = null;
        }
      };
      setTimeout(tick, STREAM_TICK_MS);
    }
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx = buffer.indexOf("\n");
      while (idx !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) {
          let evt = null;
          try {
            evt = JSON.parse(line);
          } catch (e) {
            evt = null;
          }
          if (evt && evt.type === "delta") {
            pending += evt.text || "";
            startTicker();
          } else if (evt && evt.type === "done") {
            doneEvent = evt;
            startTicker();
          } else if (evt && evt.type === "error") {
            setBubbleText(bubble, evt.message || "Request failed.");
          }
        }
        idx = buffer.indexOf("\n");
      }
    }
    if (pending.length) {
      startTicker();
      return;
    }
    if (doneEvent) {
      text = doneEvent.answer || text;
      setBubbleText(bubble, text);
      appendCitationsToBubble(bubble, doneEvent.citations || []);
      if (chat) chat.scrollTop = chat.scrollHeight;
    }
  }

  async function sendMessage() {
    const msg = (input.value || "").trim();
    if (!msg) return;
    if (!pk) {
      appendBubble("Widget is not configured (missing bot key).", "bot");
      return;
    }

    input.value = "";
    appendBubble(msg, "user");
    appendTypingBubble();
    send.disabled = true;

    try {
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, site_url: siteUrl, site_title: siteTitle }),
      });
      removeTypingBubble();
      const isStream = (resp.headers.get("content-type") || "").includes("application/x-ndjson");
      if (!resp.ok) {
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        appendBubble(data.detail || data.answer || `Error (${resp.status})`, "bot");
      } else if (isStream) {
        await streamResponse(resp);
      } else {
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        appendBubble(data.answer || "", "bot", data.citations || []);
      }
    } catch (e) {
      removeTypingBubble();
      appendBubble("Request failed: " + e.message, "bot");
    } finally {
      send.disabled = false;
    }
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
})();
