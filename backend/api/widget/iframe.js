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
    bubble.textContent = WELCOME_TEXT;
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
    div.textContent = text;
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
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, site_url: siteUrl, site_title: siteTitle }),
      });
      const isJson = (resp.headers.get("content-type") || "").includes("application/json");
      const data = isJson ? await resp.json() : { answer: await resp.text() };
      removeTypingBubble();
      if (!resp.ok) {
        appendBubble(data.detail || data.answer || `Error (${resp.status})`, "bot");
      } else {
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
