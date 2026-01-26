(() => {
  const params = new URLSearchParams(location.search);
  const pk = params.get("pk") || "";
  const apiBase = params.get("apiBase") || location.origin;
  const siteUrl = params.get("siteUrl") || document.referrer || "";
  const siteTitle = params.get("siteTitle") || "";

  const chat = document.getElementById("chat");
  const input = document.getElementById("msg");
  const send = document.getElementById("send");
  const status = document.getElementById("status");
  const crawlBtn = document.getElementById("crawl");
  const cancelBtn = document.getElementById("cancelCrawl");
  const crawlStatus = document.getElementById("crawlStatus");

  let crawlPoll = null;

  function setCrawlStatus(text) {
    if (crawlStatus) crawlStatus.textContent = text || "";
  }

  function appendBubble(text, who, citations) {
    const div = document.createElement("div");
    div.className = `bubble ${who}`;
    div.textContent = text;
    chat.appendChild(div);

    if (citations && citations.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.innerHTML =
        "<div><b>Sources</b></div>" +
        citations
          .slice(0, 6)
          .map((c) => {
            const url = (c && c.url) || "";
            if (!url) return "";
            const safe = url.replace(/"/g, "&quot;");
            return `<div><a href="${safe}" target="_blank" rel="noopener noreferrer">${safe}</a></div>`;
          })
          .join("");
      div.appendChild(meta);
    }
    chat.scrollTop = chat.scrollHeight;
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
    status.textContent = "Thinking…";
    send.disabled = true;

    try {
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, site_url: siteUrl, site_title: siteTitle }),
      });
      const isJson = (resp.headers.get("content-type") || "").includes("application/json");
      const data = isJson ? await resp.json() : { answer: await resp.text() };
      if (!resp.ok) {
        appendBubble(data.detail || data.answer || `Error (${resp.status})`, "bot");
      } else {
        appendBubble(data.answer || "", "bot", data.citations || []);
      }
    } catch (e) {
      appendBubble(`Request failed: ${e}`, "bot");
    } finally {
      status.textContent = "Ready";
      send.disabled = false;
    }
  }

  async function startCrawl() {
    if (!pk) {
      setCrawlStatus("Missing bot key");
      return;
    }
    if (!siteUrl) {
      setCrawlStatus("Missing site URL");
      return;
    }

    setCrawlStatus("Starting…");
    if (crawlBtn) crawlBtn.disabled = true;
    if (cancelBtn) cancelBtn.disabled = false;

    try {
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: siteUrl }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        setCrawlStatus(data.detail || data.message || `Error (${resp.status})`);
        if (crawlBtn) crawlBtn.disabled = false;
        if (cancelBtn) cancelBtn.disabled = true;
        return;
      }
      setCrawlStatus("Crawling…");

      if (crawlPoll) clearInterval(crawlPoll);
      crawlPoll = setInterval(async () => {
        try {
          const st = await fetch(
            `${apiBase}/v1/pk/${encodeURIComponent(pk)}/index/status?url=${encodeURIComponent(siteUrl)}`
          );
          const sd = await st.json().catch(() => ({}));
          if (!st.ok) {
            setCrawlStatus(sd.detail || `Status error (${st.status})`);
            return;
          }
          if (sd.status === "not_found") {
            setCrawlStatus("Waiting…");
            return;
          }
          const stage = sd.stage || sd.status;
          setCrawlStatus(stage || "Working…");
          if (["done", "error", "cancelled", "import_submitted"].includes(stage)) {
            clearInterval(crawlPoll);
            crawlPoll = null;
            if (crawlBtn) crawlBtn.disabled = false;
            if (cancelBtn) cancelBtn.disabled = true;
          }
        } catch (e) {
          setCrawlStatus(`Status error: ${e}`);
        }
      }, 2000);
    } catch (e) {
      setCrawlStatus(`Request failed: ${e}`);
      if (crawlBtn) crawlBtn.disabled = false;
      if (cancelBtn) cancelBtn.disabled = true;
    }
  }

  async function cancelCrawl() {
    if (!pk || !siteUrl) return;
    try {
      const resp = await fetch(`${apiBase}/v1/pk/${encodeURIComponent(pk)}/index/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: siteUrl }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        setCrawlStatus(data.detail || data.message || `Cancel error (${resp.status})`);
        return;
      }
      setCrawlStatus("Stopping…");
      if (crawlBtn) crawlBtn.disabled = false;
      if (cancelBtn) cancelBtn.disabled = true;
    } catch (e) {
      setCrawlStatus(`Cancel error: ${e}`);
    }
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
  if (crawlBtn) crawlBtn.addEventListener("click", startCrawl);
  if (cancelBtn) cancelBtn.addEventListener("click", cancelCrawl);
})();
