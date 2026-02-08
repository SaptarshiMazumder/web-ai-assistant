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
  const suggestedMessagesParam = params.get("suggestedMessages");
  const escalationsEnabled = parseBool(params.get("escalationsEnabled"), false);
  const availabilityCheckEnabled = parseBool(params.get("availabilityCheckEnabled"), false);
  const sessionKey = pk ? `webai_session_${pk}` : null;
  let sessionId = sessionKey ? localStorage.getItem(sessionKey) || "" : "";
  let isSessionEnded = false;
  let botPending = false;
  let hasBotReply = false;
  const seenMessageIds = new Set();

  function parseBool(val, def) {
    if (val == null || val === "") return def;
    const v = String(val).toLowerCase();
    return v === "true" || v === "yes" || v === "1";
  }

  function normalizeSuggestions(list) {
    if (!Array.isArray(list)) return [];
    return list
      .map((item, idx) => {
        const label = item && typeof item.label === "string" ? item.label : "";
        const type = item && (item.type === "ai_response" || item.type === "escalate" || item.type === "availability") ? item.type : "user_message";
        const message = item && typeof item.message === "string" ? item.message : "";
        const prompt = item && typeof item.prompt === "string" ? item.prompt : "";
        return { id: String(item && item.id ? item.id : `suggest_${idx}`), label, type, message, prompt };
      })
      .filter((item) => item.label);
  }

  function getInitialSuggestions() {
    if (suggestedMessagesParam) {
      try {
        const parsed = JSON.parse(suggestedMessagesParam);
        const normalized = normalizeSuggestions(parsed);
        if (normalized.length) return normalized;
      } catch (e) {}
    }
    return [
      { id: "default_1", label: "What can you do?", type: "user_message", message: "What can you do?" },
      { id: "default_2", label: "Ask a question", type: "user_message", message: "Ask a question" },
      { id: "default_3", label: "Get help", type: "user_message", message: "Get help" },
      { id: "default_4", label: "Escalate to support", type: "escalate", message: "" },
    ];
  }

  if (sessionId) {
    // session restored from storage
  }

  const SOFT_WRAP_TOKEN_MIN = 32;
  const SOFT_WRAP_CHUNK = 24;
  const SOFT_WRAP_SEPARATORS = /[\/\-\_\.\?\&\=\#\:\@]/;
  const suggestedMessages = getInitialSuggestions();

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

  function setSession(id) {
    if (!id || !sessionKey) return;
    sessionId = id;
    try {
      localStorage.setItem(sessionKey, id);
    } catch (e) {
      // ignore storage failures
    }
    updateEscalationUI();
  }

  const WELCOME_TEXT = welcomeMessage;

  const chat = document.getElementById("chat");
  const messagesEl = document.getElementById("messages");
  const welcomeEl = document.getElementById("welcome");
  const quickActionsEl = document.getElementById("quickActions");
  const input = document.getElementById("msg");
  const send = document.getElementById("send");
  const endChat = document.getElementById("endChat");
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

    // Suggested messages are rendered in the sticky quick actions area below.
  }

  function buildSuggestionButton(item) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.label;
    btn.addEventListener("click", () => {
      if (item.type === "ai_response") {
        const prompt = item.prompt || item.message || item.label;
        const display = item.label || prompt;
        sendMessageWithContent(prompt, display);
        return;
      }
      if (item.type === "escalate") {
        openEscalationModal();
        return;
      }
      if (item.type === "availability") {
        openAvailabilityModal();
        return;
      }
      const content = item.message || item.label;
      sendMessageWithContent(content, item.label || content);
    });
    return btn;
  }

  function renderQuickActions(force = false) {
    if (!quickActionsEl) return;
    quickActionsEl.innerHTML = "";
    if (!force && (isSessionEnded || botPending || !hasBotReply)) return;
    suggestedMessages.forEach((item) => {
      if (item.type === "escalate" && !escalationsEnabled) return;
      quickActionsEl.appendChild(buildSuggestionButton(item));
    });
  }

  function hideWelcome() {
    if (welcomeEl) welcomeEl.style.display = "none";
  }

  function appendTypingBubble() {
    removeTypingBubble();
    botPending = true;
    renderQuickActions();
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
  renderQuickActions(true);
  if (sessionId) {
    loadHistory();
    updateEscalationUI();
  }


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

  function friendlyLabelFromUrl(url) {
    try {
      var m = url.match(/^https?:\/\/([^\/]+)(\/[^?#]*)?/);
      if (!m) return "";
      var host = m[1] || "";
      var path = ((m[2] || "/").replace(/\/$/, "") || "/").replace(/^\//, "");
      if (path && path !== "") {
        var last = path.split("/").pop();
        var cleaned = last.replace(/-/g, " ").replace(/_/g, " ");
        return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : host;
      }
      return host;
    } catch (e) {}
    return "";
  }

  function formatCitationHtml(c) {
    const url = (c && c.url) || "";
    if (!url) return "";
    var u = url.trim().toLowerCase();
    if (u.startsWith("gs://")) return "";
    if (!u.startsWith("http://") && !u.startsWith("https://")) return "";
    const safe = url.replace(/"/g, "&quot;");
    var label = friendlyLabelFromUrl(url);
    if (!label) {
      var m = url.match(/^https?:\/\/([^\/]+)(\/[^?#]*)?/);
      label = (m && (m[2] || m[1])) ? (m[2] || m[1]).replace(/^\//, "") || m[1] : url;
    }
    return '<div><a href="' + safe + '" target="_blank" rel="noopener noreferrer">' + String(label).replace(/</g, "&lt;").replace(/>/g, "&gt;") + "</a></div>";
  }

  function appendBubble(text, who, citations, senderName) {
    hideWelcome();

    const rowDiv = document.createElement("div");
    rowDiv.className = "message-row " + (who === "user" ? "user-row" : "");

    if (who === "bot") {
      rowDiv.appendChild(botAvatarEl());
    }

    const div = document.createElement("div");
    div.className = "bubble " + who;
    if (senderName) {
      const label = document.createElement("div");
      label.className = "sender-label";
      label.textContent = senderName;
      div.appendChild(label);
    }
    setBubbleText(div, text, who);
    if (who === "user") {
      div.style.background = color;
      div.style.color = textColor;
    }
    if (citations && citations.length && displaySources) {
      const items = citations.slice(0, 6).map(formatCitationHtml).filter(Boolean);
      if (items.length) {
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML =
          "<div><b>" + (sourcesLabel.replace(/</g, "&lt;").replace(/>/g, "&gt;")) + "</b></div>" +
          items.join("");
        div.appendChild(meta);
      }
    }
    rowDiv.appendChild(div);

    if (messagesEl) messagesEl.appendChild(rowDiv);
    if (chat) chat.scrollTop = chat.scrollHeight;
    if (who === "bot") {
      hasBotReply = true;
      botPending = false;
      renderQuickActions();
    }
  }

  function openEscalationModal() {
    if (!escalationsEnabled) {
      appendBubble("Escalations are currently disabled.", "bot");
      return;
    }
    if (document.getElementById("escalation-modal")) return;
    const overlay = document.createElement("div");
    overlay.id = "escalation-modal";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(15, 23, 42, 0.6)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "9999";

    const card = document.createElement("div");
    card.style.background = theme === "dark" ? "#0f172a" : "#ffffff";
    card.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";
    card.style.borderRadius = "14px";
    card.style.padding = "18px";
    card.style.width = "90%";
    card.style.maxWidth = "320px";
    card.style.boxShadow = "0 16px 40px rgba(0,0,0,0.25)";

    const title = document.createElement("div");
    title.textContent = "Escalate to support";
    title.style.fontWeight = "600";
    title.style.marginBottom = "6px";

    const subtitle = document.createElement("div");
    subtitle.textContent = "Enter your email so support can reach you.";
    subtitle.style.fontSize = "12px";
    subtitle.style.color = theme === "dark" ? "#94a3b8" : "#64748b";
    subtitle.style.marginBottom = "12px";

    const inputEl = document.createElement("input");
    inputEl.type = "email";
    inputEl.placeholder = "you@email.com";
    inputEl.style.width = "100%";
    inputEl.style.padding = "10px 12px";
    inputEl.style.borderRadius = "10px";
    inputEl.style.border = "1px solid #e2e8f0";
    inputEl.style.background = theme === "dark" ? "#0b1220" : "#ffffff";
    inputEl.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";

    const detailsLabel = document.createElement("div");
    detailsLabel.textContent = "Details (optional)";
    detailsLabel.style.fontSize = "12px";
    detailsLabel.style.color = theme === "dark" ? "#94a3b8" : "#64748b";
    detailsLabel.style.marginTop = "10px";

    const detailsEl = document.createElement("textarea");
    detailsEl.placeholder = "Tell us a bit more about your request (optional)";
    detailsEl.rows = 3;
    detailsEl.style.width = "100%";
    detailsEl.style.padding = "10px 12px";
    detailsEl.style.borderRadius = "10px";
    detailsEl.style.border = "1px solid #e2e8f0";
    detailsEl.style.background = theme === "dark" ? "#0b1220" : "#ffffff";
    detailsEl.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";
    detailsEl.style.marginTop = "6px";

    const errorEl = document.createElement("div");
    errorEl.style.fontSize = "12px";
    errorEl.style.color = "#ef4444";
    errorEl.style.marginTop = "6px";
    errorEl.style.display = "none";

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.marginTop = "12px";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.flex = "1";
    cancelBtn.style.height = "36px";
    cancelBtn.style.borderRadius = "10px";
    cancelBtn.style.border = "1px solid #e2e8f0";
    cancelBtn.style.background = "transparent";
    cancelBtn.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";

    const submitBtn = document.createElement("button");
    submitBtn.textContent = "Submit";
    submitBtn.style.flex = "1";
    submitBtn.style.height = "36px";
    submitBtn.style.borderRadius = "10px";
    submitBtn.style.border = "0";
    submitBtn.style.background = "var(--widget-color)";
    submitBtn.style.color = "var(--widget-text-color)";

    cancelBtn.onclick = () => overlay.remove();
    submitBtn.onclick = async () => {
      const email = (inputEl.value || "").trim();
      if (!email || email.indexOf("@") === -1) {
        errorEl.textContent = "Please enter a valid email.";
        errorEl.style.display = "block";
        return;
      }
      const details = (detailsEl.value || "").trim();
      submitBtn.disabled = true;
      try {
        const targetSession = sessionId || "new";
        const resp = await fetch(
          `${apiBase}/v1/pk/${encodeURIComponent(pk)}/conversations/${encodeURIComponent(targetSession)}/escalate`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ visitor_email: email, details: details || null, site_url: siteUrl || null, site_title: siteTitle || null }),
          }
        );
        if (!resp.ok) {
          const data = await resp.json().catch(async () => ({ detail: await resp.text() }));
          errorEl.textContent = data.detail || "Failed to submit escalation.";
          errorEl.style.display = "block";
          submitBtn.disabled = false;
          return;
        }
        const data = await resp.json().catch(() => null);
        if (data && data.session_id) setSession(data.session_id);
        overlay.remove();
        appendBubble("Thanks! Support has been notified and will reach out soon.", "bot");
      } catch (e) {
        errorEl.textContent = "Failed to submit escalation.";
        errorEl.style.display = "block";
        submitBtn.disabled = false;
      }
    };

    actions.appendChild(cancelBtn);
    actions.appendChild(submitBtn);
    card.appendChild(title);
    card.appendChild(subtitle);
    card.appendChild(inputEl);
    card.appendChild(detailsLabel);
    card.appendChild(detailsEl);
    card.appendChild(errorEl);
    card.appendChild(actions);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    setTimeout(() => inputEl.focus(), 0);
  }

  function openAvailabilityModal() {
    if (document.getElementById("availability-modal")) return;

    const today = new Date();
    const checkInDefault = new Date(today);
    checkInDefault.setDate(checkInDefault.getDate() + 7);
    const checkOutDefault = new Date(checkInDefault);
    checkOutDefault.setDate(checkOutDefault.getDate() + 2);

    const fmt = (d) => d.toISOString().slice(0, 10);

    const overlay = document.createElement("div");
    overlay.id = "availability-modal";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(15, 23, 42, 0.6)";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.zIndex = "9999";

    const card = document.createElement("div");
    card.style.background = theme === "dark" ? "#0f172a" : "#ffffff";
    card.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";
    card.style.borderRadius = "14px";
    card.style.padding = "18px";
    card.style.width = "90%";
    card.style.maxWidth = "320px";
    card.style.boxShadow = "0 16px 40px rgba(0,0,0,0.25)";

    const title = document.createElement("div");
    title.textContent = "Check room availability";
    title.style.fontWeight = "600";
    title.style.marginBottom = "6px";

    const subtitle = document.createElement("div");
    subtitle.textContent = "Enter your dates and guest details.";
    subtitle.style.fontSize = "12px";
    subtitle.style.color = theme === "dark" ? "#94a3b8" : "#64748b";
    subtitle.style.marginBottom = "12px";

    const inputStyle = { width: "100%", padding: "10px 12px", borderRadius: "10px", border: "1px solid #e2e8f0", background: theme === "dark" ? "#0b1220" : "#ffffff", color: theme === "dark" ? "#e2e8f0" : "#0f172a", boxSizing: "border-box" };
    const labelStyle = { fontSize: "12px", color: theme === "dark" ? "#94a3b8" : "#64748b", marginTop: "10px", marginBottom: "4px" };

    const checkInLabel = document.createElement("div");
    checkInLabel.textContent = "Check-in";
    Object.assign(checkInLabel.style, labelStyle);
    const checkInEl = document.createElement("input");
    checkInEl.type = "date";
    checkInEl.value = fmt(checkInDefault);
    Object.assign(checkInEl.style, inputStyle);

    const checkOutLabel = document.createElement("div");
    checkOutLabel.textContent = "Check-out";
    Object.assign(checkOutLabel.style, labelStyle);
    const checkOutEl = document.createElement("input");
    checkOutEl.type = "date";
    checkOutEl.value = fmt(checkOutDefault);
    Object.assign(checkOutEl.style, inputStyle);

    const adultsLabel = document.createElement("div");
    adultsLabel.textContent = "Adults";
    Object.assign(adultsLabel.style, labelStyle);
    const adultsEl = document.createElement("input");
    adultsEl.type = "number";
    adultsEl.min = 1;
    adultsEl.value = 2;
    Object.assign(adultsEl.style, inputStyle);

    const roomsLabel = document.createElement("div");
    roomsLabel.textContent = "Rooms";
    Object.assign(roomsLabel.style, labelStyle);
    const roomsEl = document.createElement("input");
    roomsEl.type = "number";
    roomsEl.min = 1;
    roomsEl.value = 1;
    Object.assign(roomsEl.style, inputStyle);

    const errorEl = document.createElement("div");
    errorEl.style.fontSize = "12px";
    errorEl.style.color = "#ef4444";
    errorEl.style.marginTop = "6px";
    errorEl.style.display = "none";

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.marginTop = "12px";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.flex = "1";
    cancelBtn.style.height = "36px";
    cancelBtn.style.borderRadius = "10px";
    cancelBtn.style.border = "1px solid #e2e8f0";
    cancelBtn.style.background = "transparent";
    cancelBtn.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";

    const submitBtn = document.createElement("button");
    submitBtn.textContent = "Check availability";
    submitBtn.style.flex = "1";
    submitBtn.style.height = "36px";
    submitBtn.style.borderRadius = "10px";
    submitBtn.style.border = "0";
    submitBtn.style.background = "var(--widget-color)";
    submitBtn.style.color = "var(--widget-text-color)";

    cancelBtn.onclick = () => overlay.remove();
    submitBtn.onclick = () => {
      const checkIn = (checkInEl.value || "").trim();
      const checkOut = (checkOutEl.value || "").trim();
      const adults = Math.max(1, parseInt(adultsEl.value, 10) || 2);
      const rooms = Math.max(1, parseInt(roomsEl.value, 10) || 1);

      if (!checkIn || !checkOut) {
        errorEl.textContent = "Please enter check-in and check-out dates.";
        errorEl.style.display = "block";
        return;
      }
      if (new Date(checkOut) <= new Date(checkIn)) {
        errorEl.textContent = "Check-out must be after check-in.";
        errorEl.style.display = "block";
        return;
      }

      const message = "Check room availability: check-in " + checkIn + ", check-out " + checkOut + ", " + adults + " adults, " + rooms + " room" + (rooms !== 1 ? "s" : "");
      const display = "Check room availability for " + checkIn + " to " + checkOut + ", " + adults + " adults, " + rooms + " room" + (rooms !== 1 ? "s" : "");
      overlay.remove();
      sendMessageWithContent(message, display);
    };

    card.appendChild(title);
    card.appendChild(subtitle);
    card.appendChild(checkInLabel);
    card.appendChild(checkInEl);
    card.appendChild(checkOutLabel);
    card.appendChild(checkOutEl);
    card.appendChild(adultsLabel);
    card.appendChild(adultsEl);
    card.appendChild(roomsLabel);
    card.appendChild(roomsEl);
    card.appendChild(errorEl);
    card.appendChild(actions);
    actions.appendChild(cancelBtn);
    actions.appendChild(submitBtn);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    setTimeout(() => checkInEl.focus(), 0);
  }

  async function loadHistory() {
    if (!pk || !sessionId) return;
    try {
      const resp = await fetch(
        `${apiBase}/v1/pk/${encodeURIComponent(pk)}/conversations/${encodeURIComponent(sessionId)}?limit=200`
      );
      if (!resp.ok) return;
      const data = await resp.json();
      const messages = data && data.messages ? data.messages : [];
      messages.forEach((m) => {
        if (!m) return;
        const mid = m.message_id ? String(m.message_id) : "";
        if (mid && seenMessageIds.has(mid)) return;
        if (mid) seenMessageIds.add(mid);
        appendBubble(m.content || "", m.role || "bot", [], m.sender_name || "");
      });
    } catch (e) {}
  }

  function updateEscalationUI() {
    if (input) input.disabled = isSessionEnded;
    if (send) send.disabled = isSessionEnded;
    renderQuickActions();
  }

  // Strip numbered bracket citations like [1], [1, 2], [1, 11, 35, 75]
  // Negative lookbehind avoids clobbering markdown links like [text](url)
  function stripBracketCitations(text) {
    if (!text) return text;
    // Remove bracket-number references not followed by ( (which would be markdown links)
    text = text.replace(/\[[\d,\s]+\](?!\()/g, "");
    // Remove trailing bullet URL lists
    text = text.replace(/(?:^|\n)[\s]*[-*•]\s*https?:\/\/\S+.*/g, "");
    text = text.replace(/(?:^|\n)[\s]*\d+\.\s*https?:\/\/\S+.*/g, "");
    // Clean up double spaces and excess newlines
    text = text.replace(/  +/g, " ");
    text = text.replace(/\n{3,}/g, "\n\n");
    return text.trim();
  }

  function parseMarkdownLinks(text) {
    var parts = [];
    // First: parse markdown links [text](url)
    var re = /\[([^\]]*)\]\(([^)]*)\)/g;
    var last = 0;
    var m;
    while ((m = re.exec(text)) !== null) {
      var url = (m[2] || "").trim();
      if (url && (url.toLowerCase().startsWith("http://") || url.toLowerCase().startsWith("https://"))) {
        if (m.index > last) parts.push({ type: "text", content: text.slice(last, m.index) });
        parts.push({ type: "link", text: (m[1] || "").trim() || "here", url: url });
        last = m.index + m[0].length;
      }
    }
    if (last < text.length) parts.push({ type: "text", content: text.slice(last) });
    if (!parts.length) return null;
    // Second pass: find bare URLs in text segments and convert to links
    var final = [];
    var urlRe = /\bhttps?:\/\/[^\s<>\[\]"']+/g;
    parts.forEach(function (p) {
      if (p.type !== "text") { final.push(p); return; }
      var content = p.content;
      var um;
      var uLast = 0;
      while ((um = urlRe.exec(content)) !== null) {
        if (um.index > uLast) final.push({ type: "text", content: content.slice(uLast, um.index) });
        var bareUrl = um[0].replace(/[.,;:!?)]+$/, "");
        var trailingPunct = um[0].slice(bareUrl.length);
        final.push({ type: "link", text: friendlyLabelFromUrl(bareUrl) || "here", url: bareUrl });
        uLast = um.index + bareUrl.length;
        if (trailingPunct) final.push({ type: "text", content: trailingPunct });
      }
      if (uLast < content.length) final.push({ type: "text", content: content.slice(uLast) });
    });
    return final.length ? final : null;
  }

  function setBubbleText(bubble, text, who) {
    const existingLabel = bubble.querySelector(".sender-label");
    bubble.innerHTML = "";
    if (existingLabel) bubble.appendChild(existingLabel);
    var cleaned = who === "bot" && typeof text === "string" ? stripBracketCitations(text) : text;
    var parsed = who === "bot" && typeof cleaned === "string" ? parseMarkdownLinks(cleaned) : null;
    if (parsed && parsed.length > 0) {
      parsed.forEach(function (p) {
        if (p.type === "text") appendSoftWrappedText(bubble, p.content);
        else if (p.type === "link") {
          var a = document.createElement("a");
          a.href = p.url.replace(/"/g, "&quot;");
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = p.text;
          bubble.appendChild(a);
        }
      });
    } else {
      appendSoftWrappedText(bubble, cleaned || "");
    }
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
    const items = citations.slice(0, 6).map(formatCitationHtml).filter(Boolean);
    if (!items.length) return;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      "<div><b>" + (sourcesLabel.replace(/</g, "&lt;").replace(/>/g, "&gt;")) + "</b></div>" +
      items.join("");
    bubble.appendChild(meta);
  }

  const STREAM_TICK_MS = 24;
  const STREAM_CHARS_PER_TICK = 3;

  async function streamResponse(resp) {
    if (!resp.body) throw new Error("No response body");
    botPending = true;
    renderQuickActions();
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let pending = "";
    let ticking = false;
    let doneEvent = null;
    const bubble = ensureStreamingBubble();
    const headerSession = resp.headers.get("x-conversation-id");
    if (headerSession) setSession(headerSession);
    function startTicker() {
      if (ticking) return;
      ticking = true;
      const tick = () => {
        if (pending.length > 0) {
          const slice = pending.slice(0, STREAM_CHARS_PER_TICK);
          pending = pending.slice(STREAM_CHARS_PER_TICK);
          text += slice;
          setBubbleText(bubble, text, "bot");
          if (chat) chat.scrollTop = chat.scrollHeight;
          setTimeout(tick, STREAM_TICK_MS);
          return;
        }
        ticking = false;
        if (doneEvent) {
          text = doneEvent.answer || text;
          setBubbleText(bubble, text, "bot");
          appendCitationsToBubble(bubble, doneEvent.citations || []);
          if (chat) chat.scrollTop = chat.scrollHeight;
          doneEvent = null;
          botPending = false;
          hasBotReply = true;
          renderQuickActions();
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
          } else if (evt && evt.type === "meta") {
            if (evt.session_id) setSession(evt.session_id);
          } else if (evt && evt.type === "done") {
            doneEvent = evt;
            if (evt.session_id) setSession(evt.session_id);
            startTicker();
          } else if (evt && evt.type === "error") {
            setBubbleText(bubble, evt.message || "Request failed.", "bot");
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
      setBubbleText(bubble, text, "bot");
      appendCitationsToBubble(bubble, doneEvent.citations || []);
      if (chat) chat.scrollTop = chat.scrollHeight;
      botPending = false;
      hasBotReply = true;
      renderQuickActions();
    }
  }

  async function sendMessageWithContent(messageText, displayText) {
    const msg = (messageText || "").trim();
    if (!msg) return;
    if (!pk) {
      appendBubble("Widget is not configured (missing bot key).", "bot");
      return;
    }
    if (isSessionEnded) {
      appendBubble("This chat has ended.", "bot");
      return;
    }

    if (input) input.value = "";
    appendBubble(displayText || msg, "user");
    appendTypingBubble();
    send.disabled = true;

    try {
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, site_url: siteUrl, site_title: siteTitle, session_id: sessionId || undefined }),
      });
      removeTypingBubble();
      const isStream = (resp.headers.get("content-type") || "").includes("application/x-ndjson");
      if (!resp.ok) {
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        appendBubble(data.detail || data.answer || `Error (${resp.status})`, "bot");
        botPending = false;
        renderQuickActions();
      } else if (isStream) {
        await streamResponse(resp);
      } else {
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        if (data && data.session_id) setSession(data.session_id);
        appendBubble(data.answer || "", "bot", data.citations || []);
        botPending = false;
        renderQuickActions();
      }
    } catch (e) {
      removeTypingBubble();
      appendBubble("Request failed: " + e.message, "bot");
      botPending = false;
      renderQuickActions();
    } finally {
      send.disabled = false;
    }
  }

  async function endChatSession() {
    if (!pk || !sessionId) return;
    try {
      await fetch(
        `${apiBase}/v1/pk/${encodeURIComponent(pk)}/conversations/${encodeURIComponent(sessionId)}/end`,
        {
          method: "POST",
        }
      );
    } catch (e) {}
    isSessionEnded = true;
    if (messagesEl) messagesEl.innerHTML = "";
    if (welcomeEl) welcomeEl.innerHTML = "";
    appendBubble("Chat ended.", "bot");
    if (sessionKey) {
      try {
        localStorage.removeItem(sessionKey);
      } catch (e) {}
    }
    sessionId = "";
    hasBotReply = false;
    botPending = false;
    updateEscalationUI();
    renderQuickActions(true);
  }

  async function sendMessage() {
    const msg = (input.value || "").trim();
    if (!msg) return;
    await sendMessageWithContent(msg, msg);
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
  if (endChat) {
    endChat.addEventListener("click", endChatSession);
  }

  // Do not auto-end on reload; session ends via inactivity or explicit end.
})();
