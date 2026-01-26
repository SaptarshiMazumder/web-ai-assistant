// Embeddable loader script.
// Usage:
// <script async src="https://YOUR_API_HOST/widget/widget.js" data-bot-key="pk_..." data-api-base="https://YOUR_API_HOST"></script>
(function () {
  const script = document.currentScript;
  if (!script) return;

  const botKey = script.getAttribute("data-bot-key") || script.getAttribute("data-bot-id") || "";
  const apiBase = script.getAttribute("data-api-base") || (new URL(script.src)).origin;

  if (!botKey) {
    // Silent: avoid breaking host pages.
    return;
  }

  // Avoid double-mount.
  if (window.__WEB_AI_WIDGET_MOUNTED__) return;
  window.__WEB_AI_WIDGET_MOUNTED__ = true;

  const iframe = document.createElement("iframe");
  const siteUrl = window.location && window.location.href ? window.location.href : "";
  const siteTitle = document && document.title ? document.title : "";
  iframe.src = `${apiBase}/widget/iframe.html?pk=${encodeURIComponent(botKey)}&apiBase=${encodeURIComponent(apiBase)}&siteUrl=${encodeURIComponent(siteUrl)}&siteTitle=${encodeURIComponent(siteTitle)}`;
  iframe.title = "Chatbot";
  iframe.style.position = "fixed";
  iframe.style.bottom = "16px";
  iframe.style.right = "16px";
  iframe.style.width = "380px";
  iframe.style.height = "560px";
  iframe.style.border = "0";
  iframe.style.zIndex = "2147483647";
  iframe.style.boxShadow = "0 16px 40px rgba(0,0,0,0.22)";
  iframe.style.borderRadius = "12px";
  iframe.style.background = "transparent";

  document.body.appendChild(iframe);
})();
