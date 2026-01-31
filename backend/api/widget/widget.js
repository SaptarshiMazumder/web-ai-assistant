// Embeddable loader script.
// Fetches widget config from API (stored per bot in DB), then merges with any data-* on script tag (overrides).
// Snippet can be minimal: <script async src=".../widget.js" data-bot-key="pk_..." data-api-base="..."></script>
(function () {
  const script = document.currentScript;
  if (!script) return;

  const botKey = script.getAttribute("data-bot-key") || script.getAttribute("data-bot-id") || "";
  const apiBase = script.getAttribute("data-api-base") || (new URL(script.src)).origin;

  if (!botKey) {
    return;
  }

  if (window.__WEB_AI_WIDGET_MOUNTED__) return;
  window.__WEB_AI_WIDGET_MOUNTED__ = true;

  const siteUrl = window.location && window.location.href ? window.location.href : "";
  const siteTitle = document && document.title ? document.title : "";
  const configUrl = apiBase + "/v1/pk/" + encodeURIComponent(botKey) + "/widget-config";
  var attrToParam = {
    "data-position": "position", "data-color": "color", "data-title": "title", "data-size": "size",
    "data-placeholder": "placeholder", "data-footer": "footer", "data-welcome-message": "welcomeMessage", "data-theme": "theme", "data-text-color": "textColor",
    "data-launcher-icon": "launcherIcon", "data-launcher-text": "launcherText", "data-header-icon": "headerIcon", "data-share-icon": "shareIcon",
    "data-max-height": "maxHeight", "data-font-size": "fontSize", "data-header-size": "headerSize",
    "data-auto-popup": "autoPopup", "data-auto-scroll": "autoScroll", "data-display-sources": "displaySources",
    "data-sources-label": "sourcesLabel"
  };

  function scriptOverrides() {
    var o = {};
    for (var i = 0; i < script.attributes.length; i++) {
      var a = script.attributes[i];
      var param = attrToParam[a.name];
      if (param && a.value) o[param] = a.value;
    }
    return o;
  }

  function applyConfig(config) {
    var overrides = scriptOverrides();
    var merged = {};
    for (var k in config) if (config.hasOwnProperty(k) && config[k] != null) merged[k] = config[k];
    for (var k in overrides) if (overrides.hasOwnProperty(k)) merged[k] = overrides[k];

    var params = new URLSearchParams();
    params.set("pk", botKey);
    params.set("apiBase", apiBase);
    params.set("siteUrl", siteUrl);
    params.set("siteTitle", siteTitle);
    for (var key in merged) if (merged.hasOwnProperty(key) && merged[key] !== "") {
      params.set(key, String(merged[key]));
    }

    var position = (merged.position || "bottom-right").toLowerCase();
    var isLeft = position === "bottom-left";
    var sizeMap = { small: [320, 420], medium: [380, 560], large: [440, 640] };
    var sizeKey = (merged.size || "medium").toLowerCase();
    var dims = sizeMap[sizeKey] || sizeMap.medium;
    var maxH = parseInt(merged.maxHeight || "", 10);
    var height = (maxH >= 400 && maxH <= 800) ? maxH : dims[1];

    var iframe = document.createElement("iframe");
    iframe.src = apiBase + "/widget/iframe.html?" + params.toString();
    iframe.title = merged.title || "Chat";
    iframe.style.position = "fixed";
    iframe.style.bottom = "16px";
    iframe.style[isLeft ? "left" : "right"] = "16px";
    iframe.style.width = dims[0] + "px";
    iframe.style.height = height + "px";
    iframe.style.border = "0";
    iframe.style.zIndex = "2147483647";
    iframe.style.boxShadow = "0 16px 40px rgba(0,0,0,0.22)";
    iframe.style.borderRadius = "12px";
    iframe.style.background = "transparent";
    document.body.appendChild(iframe);
  }

  fetch(configUrl)
    .then(function (r) { return r.ok ? r.json() : {}; })
    .then(function (config) { applyConfig(config || {}); })
    .catch(function () { applyConfig(scriptOverrides()); });
})();
