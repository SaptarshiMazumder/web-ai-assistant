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
  const supportMessagesParam = params.get("supportMessages");
  const supportUi = (() => {
    let parsed = {};
    if (supportMessagesParam) {
      try {
        parsed = JSON.parse(supportMessagesParam) || {};
      } catch (e) {
        parsed = {};
      }
    }
    return {
      requested: typeof parsed.requested === "string" ? parsed.requested : "",
      disabled: typeof parsed.disabled === "string" ? parsed.disabled : "",
      modalTitle: typeof parsed.modalTitle === "string" ? parsed.modalTitle : "",
      modalSubtitle: typeof parsed.modalSubtitle === "string" ? parsed.modalSubtitle : "",
      emailPlaceholder: typeof parsed.emailPlaceholder === "string" ? parsed.emailPlaceholder : "",
      detailsLabel: typeof parsed.detailsLabel === "string" ? parsed.detailsLabel : "",
      detailsPlaceholder: typeof parsed.detailsPlaceholder === "string" ? parsed.detailsPlaceholder : "",
      cancelButton: typeof parsed.cancelButton === "string" ? parsed.cancelButton : "",
      submitButton: typeof parsed.submitButton === "string" ? parsed.submitButton : "",
      invalidEmail: typeof parsed.invalidEmail === "string" ? parsed.invalidEmail : "",
      submitFailed: typeof parsed.submitFailed === "string" ? parsed.submitFailed : "",
      submitSuccess: typeof parsed.submitSuccess === "string" ? parsed.submitSuccess : "",
    };
  })();
  const displaySources = parseBool(params.get("displaySources"), false);
  const sourcesLabel = params.get("sourcesLabel") || "Sources";
  const suggestedMessagesParam = params.get("suggestedMessages");
  const escalationsEnabled = true;
  const availabilityCheckEnabled = parseBool(params.get("availabilityCheckEnabled"), false);
  const sessionKey = pk ? `webai_session_${pk}` : null;
  let sessionId = sessionKey ? localStorage.getItem(sessionKey) || "" : "";
  let isSessionEnded = false;
  let botPending = false;
  let hasBotReply = false;
  let supportRequestSubmitted = false;
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
        const type = item && (item.type === "escalate" ? "escalate" : item.type === "show_menu" ? "show_menu" : "ai_response");
        const message = item && typeof item.message === "string" ? item.message : "";
        const prompt = item && typeof item.prompt === "string" ? item.prompt : "";
        const urls = Array.isArray(item && item.urls)
          ? item.urls.map((u) => (typeof u === "string" ? u.trim() : "")).filter(Boolean)
          : [];
        return { id: String(item && item.id ? item.id : `suggest_${idx}`), label, type, message, prompt, urls };
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
    return [];
  }

  if (sessionId) {
    // session restored from storage
  }

  const SOFT_WRAP_TOKEN_MIN = 32;
  const SOFT_WRAP_CHUNK = 24;
  const SOFT_WRAP_SEPARATORS = /[\/\-\_\.\?\&\=\#\:\@]/;
  let suggestedMessages = getInitialSuggestions();

  function updateSuggestedMessages(next) {
    if (!Array.isArray(next)) return;
    suggestedMessages = normalizeSuggestions(next);
  }

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
    const isSupportDone = item.type === "escalate" && supportRequestSubmitted;
    btn.textContent = isSupportDone ? `\u2713 ${supportUi.requested || item.label}` : item.label;
    if (isSupportDone) {
      btn.disabled = true;
      btn.classList.add("suggestion-complete");
      btn.setAttribute("aria-label", supportUi.requested || item.label);
      btn.title = supportUi.requested || item.label;
    }
    btn.addEventListener("click", () => {
      if (isSupportDone) return;
      if (item.type === "ai_response") {
        const promptBase = item.prompt || item.message || item.label;
        const display = item.label || promptBase;
        sendMessageWithContent(promptBase, display, { suggestedMessageId: item.id });
        return;
      }
      if (item.type === "escalate") {
        openEscalationModal();
        return;
      }
      if (item.type === "show_menu") {
        sendMessageWithContent(item.label || "Menu", item.label || "Menu", { suggestedMessageId: item.id });
        return;
      }
      const content = item.message || item.label;
      sendMessageWithContent(content, item.label || content, { suggestedMessageId: item.id });
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

  function isGenericLinkText(label) {
    var s = String(label || "").trim().toLowerCase().replace(/\s+/g, " ");
    if (!s) return true;
    if (/^sources?$/.test(s) || s === "link" || s === "here") return true;
    if (s.indexOf("http://") >= 0 || s.indexOf("https://") >= 0) return true;
    if (s.charAt(0) === "/" && s.indexOf(" ") === -1) return true;
    if (/^[a-z0-9.-]+\.[a-z]{2,}(\/.*)?$/.test(s) && s.indexOf(" ") === -1) return true;
    return false;
  }

  function isUrlChar(ch) {
    return !!ch && ch.charCodeAt(0) <= 127 && /[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]/.test(ch);
  }

  function countChar(text, target) {
    var count = 0;
    for (var i = 0; i < text.length; i += 1) {
      if (text.charAt(i) === target) count += 1;
    }
    return count;
  }

  function trimUrlSuffix(rawToken) {
    var value = String(rawToken || "");
    var trailing = "";
    while (value) {
      var tail = value.charAt(value.length - 1);
      if (/[.,;:!?]/.test(tail) || tail === '"' || tail === "'") {
        trailing = tail + trailing;
        value = value.slice(0, -1);
        continue;
      }
      var opener = tail === ")" ? "(" : tail === "]" ? "[" : tail === "}" ? "{" : "";
      if (opener && countChar(value, opener) < countChar(value, tail)) {
        trailing = tail + trailing;
        value = value.slice(0, -1);
        continue;
      }
      break;
    }
    return { url: value, trailing: trailing };
  }

  function isValidHttpUrl(url) {
    try {
      var parsed = new URL(String(url || ""));
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch (e) {}
    return false;
  }

  function pushTextPart(parts, content) {
    if (!content) return;
    var last = parts.length ? parts[parts.length - 1] : null;
    if (last && last.type === "text") {
      last.content += content;
      return;
    }
    parts.push({ type: "text", content: content });
  }

  function parseMarkdownLinkAt(text, start) {
    if (start < 0 || start >= text.length || text.charAt(start) !== "[") return null;
    var closeBracket = text.indexOf("]", start + 1);
    if (closeBracket < 0 || closeBracket + 1 >= text.length || text.charAt(closeBracket + 1) !== "(") return null;
    var depth = 1;
    for (var i = closeBracket + 2; i < text.length; i += 1) {
      var ch = text.charAt(i);
      if (ch === "(") depth += 1;
      else if (ch === ")") {
        depth -= 1;
        if (depth === 0) {
          return {
            start: start,
            end: i + 1,
            label: text.slice(start + 1, closeBracket),
            url: text.slice(closeBracket + 2, i),
          };
        }
      }
    }
    return null;
  }

  function findNextMarkdownLink(text, fromIndex) {
    var cursor = fromIndex || 0;
    while (cursor < text.length) {
      var openBracket = text.indexOf("[", cursor);
      if (openBracket < 0) return null;
      var match = parseMarkdownLinkAt(text, openBracket);
      if (match) return match;
      cursor = openBracket + 1;
    }
    return null;
  }

  function consumeUrlToken(text, start, explicitLabel) {
    var end = start;
    while (end < text.length && isUrlChar(text.charAt(end))) end += 1;
    if (end <= start) return null;
    var rawToken = text.slice(start, end);
    var trimmed = trimUrlSuffix(rawToken);
    var url = trimmed.url;
    if (!isValidHttpUrl(url)) return null;
    var label = String(explicitLabel || "").trim();
    if (!label || isGenericLinkText(label)) {
      label = friendlyLabelFromUrl(url) || "this page";
    }
    return {
      end: end,
      link: { type: "link", text: label, url: url },
      trailing: trimmed.trailing,
    };
  }

  function appendParsedText(parts, content) {
    var cursor = 0;
    while (cursor < content.length) {
      var match = /https?:\/\//i.exec(content.slice(cursor));
      if (!match) {
        pushTextPart(parts, content.slice(cursor));
        return;
      }
      var urlStart = cursor + match.index;
      if (urlStart > cursor) pushTextPart(parts, content.slice(cursor, urlStart));
      var consumed = consumeUrlToken(content, urlStart, "");
      if (!consumed) {
        pushTextPart(parts, content.slice(urlStart, urlStart + 1));
        cursor = urlStart + 1;
        continue;
      }
      if (!isPdfLocalUrl(consumed.link.url)) {
        parts.push(consumed.link);
      }
      if (consumed.trailing) pushTextPart(parts, consumed.trailing);
      cursor = consumed.end;
    }
  }

  function getHostname(url) {
    try {
      var m = String(url || "").trim().match(/^https?:\/\/([^\/?#]+)/i);
      if (!m) return "";
      return (m[1] || "").toLowerCase().split(":")[0];
    } catch (e) {}
    return "";
  }

  function isPdfLocalUrl(url) {
    return getHostname(url) === "pdf.local";
  }

  function formatCitationHtml(c) {
    const url = (c && c.url) || "";
    if (!url) return "";
    var u = url.trim().toLowerCase();
    if (u.startsWith("gs://")) return "";
    if (isPdfLocalUrl(url)) return "";
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
      appendBubble(supportUi.disabled, "bot");
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
    title.textContent = supportUi.modalTitle;
    title.style.fontWeight = "600";
    title.style.marginBottom = "6px";

    const subtitle = document.createElement("div");
    subtitle.textContent = supportUi.modalSubtitle;
    subtitle.style.fontSize = "12px";
    subtitle.style.color = theme === "dark" ? "#94a3b8" : "#64748b";
    subtitle.style.marginBottom = "12px";

    const inputEl = document.createElement("input");
    inputEl.type = "email";
    inputEl.placeholder = supportUi.emailPlaceholder;
    inputEl.style.width = "100%";
    inputEl.style.padding = "10px 12px";
    inputEl.style.borderRadius = "10px";
    inputEl.style.border = "1px solid #e2e8f0";
    inputEl.style.background = theme === "dark" ? "#0b1220" : "#ffffff";
    inputEl.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";

    const detailsLabel = document.createElement("div");
    detailsLabel.textContent = supportUi.detailsLabel;
    detailsLabel.style.fontSize = "12px";
    detailsLabel.style.color = theme === "dark" ? "#94a3b8" : "#64748b";
    detailsLabel.style.marginTop = "10px";

    const detailsEl = document.createElement("textarea");
    detailsEl.placeholder = supportUi.detailsPlaceholder;
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
    cancelBtn.textContent = supportUi.cancelButton;
    cancelBtn.style.flex = "1";
    cancelBtn.style.height = "36px";
    cancelBtn.style.borderRadius = "10px";
    cancelBtn.style.border = "1px solid #e2e8f0";
    cancelBtn.style.background = "transparent";
    cancelBtn.style.color = theme === "dark" ? "#e2e8f0" : "#0f172a";

    const submitBtn = document.createElement("button");
    submitBtn.textContent = supportUi.submitButton;
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
        errorEl.textContent = supportUi.invalidEmail;
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
          errorEl.textContent = data.detail || supportUi.submitFailed;
          errorEl.style.display = "block";
          submitBtn.disabled = false;
          return;
        }
        const data = await resp.json().catch(() => null);
        if (data && data.session_id) setSession(data.session_id);
        supportRequestSubmitted = true;
        renderQuickActions(true);
        overlay.remove();
        appendBubble(supportUi.submitSuccess, "bot");
      } catch (e) {
        errorEl.textContent = supportUi.submitFailed;
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
    // Remove ugly PDF/page bracket citations like: [Some_File.pdf page 1], [Document page 5]
    // (but do NOT clobber markdown links like [text](url)).
    text = text.replace(/\[[^\]]*\.pdf[^\]]*\](?!\()/gi, "");
    text = text.replace(/\[[^\]]*\bpage\s*\d+[^\]]*\](?!\()/gi, "");
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
    var cursor = 0;
    while (cursor < text.length) {
      var markdown = findNextMarkdownLink(text, cursor);
      if (!markdown) {
        appendParsedText(parts, text.slice(cursor));
        break;
      }
      if (markdown.start > cursor) appendParsedText(parts, text.slice(cursor, markdown.start));
      var consumed = consumeUrlToken(markdown.url.trim(), 0, markdown.label);
      if (!consumed) {
        pushTextPart(parts, text.slice(markdown.start, markdown.end));
      } else {
        if (!isPdfLocalUrl(consumed.link.url)) {
          parts.push(consumed.link);
        }
        var suffixText = (consumed.trailing || "") + markdown.url.trim().slice(consumed.end);
        if (suffixText) pushTextPart(parts, suffixText);
      }
      cursor = markdown.end;
    }
    return parts.length ? parts : null;
  }

  function getVisibleStreamingText(text) {
    if (!text) return text;
    // Hide incomplete URL at end
    var lower = text.toLowerCase();
    var lastHttp = Math.max(lower.lastIndexOf("https://"), lower.lastIndexOf("http://"));
    if (lastHttp >= 0) {
      var consumed = consumeUrlToken(text, lastHttp, "");
      if (consumed && consumed.end === text.length && !consumed.trailing) {
        text = text.slice(0, lastHttp);
      }
    }
    // Hide incomplete bold/italic markers at end
    // Match trailing *<text without closing *> at end of string
    var trailingStars = text.match(/(\*{1,2})([^*]{0,80})$/);
    if (trailingStars) {
      var stars = trailingStars[1];
      var rest = trailingStars[2];
      // Check if the marker is unclosed
      if (rest.indexOf(stars) === -1) {
        text = text.slice(0, text.length - trailingStars[0].length);
      }
    }
    // Hide incomplete code fence at end
    var fenceCount = (text.match(/```/g) || []).length;
    if (fenceCount % 2 !== 0) {
      text = text.slice(0, text.lastIndexOf("```"));
    }
    // Hide incomplete inline code at end
    var backtickCount = (text.match(/`/g) || []).length;
    // Subtract backticks that are part of code fences (already handled)
    var fenceTicks = (text.match(/```/g) || []).length * 3;
    var singleTicks = backtickCount - fenceTicks;
    if (singleTicks % 2 !== 0) {
      // Find the last lone backtick
      var lastBt = text.lastIndexOf("`");
      if (lastBt >= 0 && text.slice(Math.max(0, lastBt - 2), lastBt + 3).indexOf("```") === -1) {
        text = text.slice(0, lastBt);
      }
    }
    return text;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdownToHTML(text) {
    if (!text) return "";
    var escaped = escapeHtml(text);

    // Code blocks: ```...```
    var codeBlocks = [];
    escaped = escaped.replace(/```[\s\S]*?```/g, function (match) {
      var code = match.slice(3, -3).replace(/^\s*\n/, "").replace(/\n\s*$/, "");
      codeBlocks.push("<pre><code>" + code + "</code></pre>");
      return "\x00CB" + (codeBlocks.length - 1) + "\x00";
    });

    // Inline code: `...`
    var inlineCodes = [];
    escaped = escaped.replace(/`([^`\n]+)`/g, function (_, code) {
      inlineCodes.push("<code>" + code + "</code>");
      return "\x00IC" + (inlineCodes.length - 1) + "\x00";
    });

    // Bold: **text**
    escaped = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic: *text* (not inside bold)
    escaped = escaped.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

    // Headers: ## heading → bold line
    escaped = escaped.replace(/^(#{1,6})\s+(.+)$/gm, function (_, hashes, content) {
      return "<strong>" + content + "</strong>";
    });

    // Process lines for lists and paragraphs
    var lines = escaped.split("\n");
    var html = "";
    var inUl = false;
    var inOl = false;

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      // Check for placeholder (code block) - render as-is
      var cbMatch = line.match(/^\x00CB(\d+)\x00$/);
      if (cbMatch) {
        if (inUl) { html += "</ul>"; inUl = false; }
        if (inOl) { html += "</ol>"; inOl = false; }
        html += codeBlocks[parseInt(cbMatch[1])];
        continue;
      }

      // Bullet list item: - item or * item (but not bold **)
      var ulMatch = line.match(/^\s*[-*]\s+(.+)/);
      if (ulMatch && !/^\s*\*\*/.test(line)) {
        if (inOl) { html += "</ol>"; inOl = false; }
        if (!inUl) { html += "<ul>"; inUl = true; }
        html += "<li>" + ulMatch[1] + "</li>";
        continue;
      }

      // Numbered list item: 1. item
      var olMatch = line.match(/^\s*\d+\.\s+(.+)/);
      if (olMatch) {
        if (inUl) { html += "</ul>"; inUl = false; }
        if (!inOl) { html += "<ol>"; inOl = true; }
        html += "<li>" + olMatch[1] + "</li>";
        continue;
      }

      // Non-list line: close any open lists
      if (inUl) { html += "</ul>"; inUl = false; }
      if (inOl) { html += "</ol>"; inOl = false; }

      // Empty line → paragraph break
      if (line.trim() === "") {
        html += "<br>";
      } else {
        html += (html && !html.endsWith("<br>") && !html.endsWith("</ul>") && !html.endsWith("</ol>") && !html.endsWith("</pre>") ? "<br>" : "") + line;
      }
    }
    if (inUl) html += "</ul>";
    if (inOl) html += "</ol>";

    // Restore inline code placeholders
    html = html.replace(/\x00IC(\d+)\x00/g, function (_, idx) {
      return inlineCodes[parseInt(idx)];
    });

    // Restore code block placeholders (for inline occurrences)
    html = html.replace(/\x00CB(\d+)\x00/g, function (_, idx) {
      return codeBlocks[parseInt(idx)];
    });

    return html;
  }

  function setBubbleText(bubble, text, who, options) {
    var opts = options || {};
    const existingLabel = bubble.querySelector(".sender-label");
    bubble.innerHTML = "";
    if (existingLabel) bubble.appendChild(existingLabel);

    if (opts.raw === true) {
      appendSoftWrappedText(bubble, text || "");
      return;
    }

    var cleaned = who === "bot" && typeof text === "string" ? stripBracketCitations(text) : text;

    // User messages: plain text
    if (who !== "bot") {
      appendSoftWrappedText(bubble, cleaned || "");
      return;
    }

    // Bot messages: markdown rendering
    var parsed = typeof cleaned === "string" ? parseMarkdownLinks(cleaned) : null;
    if (parsed && parsed.length > 0) {
      var htmlParts = [];
      parsed.forEach(function (p) {
        if (p.type === "text") {
          htmlParts.push(renderMarkdownToHTML(p.content));
        } else if (p.type === "link") {
          htmlParts.push('<a href="' + escapeHtml(p.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(p.text) + '</a>');
        }
      });
      var labelHtml = existingLabel ? existingLabel.outerHTML : "";
      bubble.innerHTML = labelHtml + htmlParts.join("");
    } else {
      bubble.innerHTML = (existingLabel ? existingLabel.outerHTML : "") + renderMarkdownToHTML(cleaned || "");
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

  function appendAssetCarouselRow(assets) {
    if (!assets || !assets.length) return;
    hideWelcome();

    var rowDiv = document.createElement("div");
    rowDiv.className = "message-row asset-row";
    rowDiv.appendChild(botAvatarEl());

    var container = document.createElement("div");
    container.className = "asset-carousel-wrap";

    var carousel = document.createElement("div");
    carousel.className = "asset-carousel";

    var track = document.createElement("div");
    track.className = "asset-cards";

    assets.forEach(function (a) {
      var card = document.createElement("div");
      card.className = "asset-card";
      if (a.link_url) {
        card.onclick = function () { window.open(a.link_url, "_blank", "noopener"); };
      }
      var img = document.createElement("img");
      img.src = a.image_url || "";
      img.alt = a.name || "";
      img.loading = "lazy";
      img.onerror = function () { img.style.display = "none"; };
      card.appendChild(img);
      var body = document.createElement("div");
      body.className = "asset-card-body";
      var title = document.createElement("span");
      title.className = "asset-card-title";
      title.textContent = a.name || "";
      body.appendChild(title);
      if (a.description) {
        var desc = document.createElement("span");
        desc.className = "asset-card-desc";
        desc.textContent = a.description;
        body.appendChild(desc);
      }
      if (a.link_url) {
        var link = document.createElement("a");
        link.className = "asset-card-link";
        link.href = a.link_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "View \u2192";
        link.onclick = function (e) { e.stopPropagation(); };
        body.appendChild(link);
      }
      card.appendChild(body);
      track.appendChild(card);
    });

    carousel.appendChild(track);

    // Nav buttons (only if more than 1 card)
    if (assets.length > 1) {
      var prevBtn = document.createElement("button");
      prevBtn.className = "carousel-btn prev";
      prevBtn.innerHTML = '<svg viewBox="0 0 24 24"><polyline points="15 18 9 12 15 6"/></svg>';
      prevBtn.onclick = function (e) { e.stopPropagation(); track.scrollBy({ left: -210, behavior: "smooth" }); };

      var nextBtn = document.createElement("button");
      nextBtn.className = "carousel-btn next";
      nextBtn.innerHTML = '<svg viewBox="0 0 24 24"><polyline points="9 6 15 12 9 18"/></svg>';
      nextBtn.onclick = function (e) { e.stopPropagation(); track.scrollBy({ left: 210, behavior: "smooth" }); };

      carousel.appendChild(prevBtn);
      carousel.appendChild(nextBtn);

      // Dots
      var dots = document.createElement("div");
      dots.className = "carousel-dots";
      var dotEls = [];
      assets.forEach(function (_a, i) {
        var dot = document.createElement("button");
        dot.className = "carousel-dot" + (i === 0 ? " active" : "");
        dot.onclick = function (e) {
          e.stopPropagation();
          var cards = track.children;
          if (cards[i]) cards[i].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "start" });
        };
        dotEls.push(dot);
        dots.appendChild(dot);
      });
      carousel.appendChild(dots);

      // Update dots + button state on scroll
      var updateControls = function () {
        var sl = track.scrollLeft;
        var maxScroll = track.scrollWidth - track.clientWidth;
        prevBtn.disabled = sl <= 2;
        nextBtn.disabled = sl >= maxScroll - 2;
        var cardW = 210; // card width + gap
        var activeIdx = Math.round(sl / cardW);
        dotEls.forEach(function (d, i) {
          d.classList.toggle("active", i === activeIdx);
        });
      };
      track.addEventListener("scroll", updateControls, { passive: true });
      // Initial state
      setTimeout(updateControls, 50);
    }

    container.appendChild(carousel);
    rowDiv.appendChild(container);
    if (messagesEl) messagesEl.appendChild(rowDiv);
    if (chat) chat.scrollTop = chat.scrollHeight;
  }

  async function streamResponse(resp) {
    if (!resp.body) throw new Error("No response body");
    botPending = true;
    renderQuickActions();
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let doneEvent = null;
    let bubble = null;
    let finalized = false;
    function ensureBubble() {
      if (bubble) return bubble;
      removeTypingBubble();
      bubble = ensureStreamingBubble();
      return bubble;
    }
    function finalizeDoneEvent() {
      if (finalized || !doneEvent) return;
      finalized = true;
      var evt = doneEvent;
      doneEvent = null;
      var finalBubble = ensureBubble();
      text = evt.answer || text;
      setBubbleText(finalBubble, text, "bot");
      appendCitationsToBubble(finalBubble, evt.citations || []);
      appendAssetCarouselRow(evt.assets || []);
      if (chat) chat.scrollTop = chat.scrollHeight;
      botPending = false;
      hasBotReply = true;
      renderQuickActions();
    }
    const headerSession = resp.headers.get("x-conversation-id");
    if (headerSession) setSession(headerSession);

    function appendDelta(deltaText) {
      if (!deltaText) return;
      text += deltaText;
      setBubbleText(ensureBubble(), text, "bot", { raw: true });
      if (chat) chat.scrollTop = chat.scrollHeight;
    }

    function handleStreamEvent(evt) {
      if (!evt) return;
      if (evt.type === "delta") {
        appendDelta(evt.text || "");
      } else if (evt.type === "meta") {
        if (evt.session_id) setSession(evt.session_id);
      } else if (evt.type === "done") {
        doneEvent = evt;
        updateSuggestedMessages(evt.suggested_messages || evt.suggestedMessages);
        if (evt.session_id) setSession(evt.session_id);
        finalizeDoneEvent();
      } else if (evt.type === "error") {
        setBubbleText(ensureBubble(), evt.message || "Request failed.", "bot");
      }
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
          handleStreamEvent(evt);
        }
        idx = buffer.indexOf("\n");
      }
    }

    const tail = buffer.trim();
    if (tail) {
      try {
        handleStreamEvent(JSON.parse(tail));
      } catch (e) {
        // Ignore partial tail chunks.
      }
    }

    if (doneEvent && !finalized) {
      finalizeDoneEvent();
    }
  }

  async function sendMessageWithContent(messageText, displayText, options) {
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
          body: JSON.stringify({
            message: msg,
            site_url: siteUrl,
            site_title: siteTitle,
            session_id: sessionId || undefined,
            suggested_message_id: options && options.suggestedMessageId ? options.suggestedMessageId : undefined,
          }),
        });
      const isStream = (resp.headers.get("content-type") || "").includes("application/x-ndjson");
      if (!resp.ok) {
        removeTypingBubble();
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        appendBubble(data.detail || data.answer || `Error (${resp.status})`, "bot");
        botPending = false;
        renderQuickActions();
      } else if (isStream) {
        await streamResponse(resp);
      } else {
        removeTypingBubble();
        const data = await resp.json().catch(async () => ({ answer: await resp.text() }));
        if (data && data.session_id) setSession(data.session_id);
        updateSuggestedMessages(data && (data.suggested_messages || data.suggestedMessages));
        appendBubble(data.answer || "", "bot", data.citations || []);
        appendAssetCarouselRow(data && (data.assets || data.asset_cards || []));
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
    supportRequestSubmitted = false;
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

  var closeWidgetBtn = document.getElementById("closeWidget");
  if (closeWidgetBtn) {
    closeWidgetBtn.addEventListener("click", function () {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "webai-widget-close" }, "*");
      }
    });
  }

  // Do not auto-end on reload; session ends via inactivity or explicit end.
})();
