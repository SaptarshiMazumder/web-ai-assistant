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
    var availabilityCheckEnabled = merged.businessType === "hotel" && merged.allowRealtimeAvailability === true;
    params.set("availabilityCheckEnabled", availabilityCheckEnabled ? "true" : "false");
    for (var key in merged) if (merged.hasOwnProperty(key) && merged[key] !== "") {
      if (key === "suggestedMessages" && Array.isArray(merged[key])) {
        try {
          params.set("suggestedMessages", JSON.stringify(merged[key]));
        } catch (e) {}
      } else if (key === "supportMessages" && merged[key] && typeof merged[key] === "object") {
        try {
          params.set("supportMessages", JSON.stringify(merged[key]));
        } catch (e) {}
      } else {
        params.set(key, String(merged[key]));
      }
    }

    var position = (merged.position || "bottom-right").toLowerCase();
    var isLeft = position === "bottom-left";
    var sizeMap = { small: [320, 420], medium: [380, 560], large: [440, 640] };
    var sizeKey = (merged.size || "medium").toLowerCase();
    var dims = sizeMap[sizeKey] || sizeMap.medium;
    var maxH = parseInt(merged.maxHeight || "", 10);
    var height = (maxH >= 400 && maxH <= 800) ? maxH : dims[1];

    // --- Launcher button ---
    var widgetColor = merged.color || "#1976d2";
    var widgetTextColor = merged.textColor || "#ffffff";
    var launcher = document.createElement("div");
    launcher.id = "__web_ai_launcher__";
    launcher.style.cssText = "position:fixed;bottom:16px;" + (isLeft ? "left" : "right") + ":16px;" +
      "width:56px;height:56px;border-radius:50%;background:" + widgetColor + ";color:" + widgetTextColor + ";" +
      "display:flex;align-items:center;justify-content:center;cursor:pointer;" +
      "z-index:2147483647;box-shadow:0 4px 20px rgba(0,0,0,0.18);" +
      "transition:transform 0.2s ease,box-shadow 0.2s ease;";
    var launcherIcon = merged.launcherIcon;
    if (launcherIcon) {
      var img = document.createElement("img");
      img.src = launcherIcon;
      img.alt = merged.launcherText || "Chat";
      img.style.cssText = "width:28px;height:28px;object-fit:contain;";
      launcher.appendChild(img);
    } else {
      launcher.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>';
    }
    if (merged.launcherText) launcher.title = merged.launcherText;
    launcher.addEventListener("mouseenter", function () { launcher.style.transform = "scale(1.08)"; });
    launcher.addEventListener("mouseleave", function () { launcher.style.transform = "scale(1)"; });

    // --- Chat iframe ---
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
    iframe.style.display = "none"; // Start hidden

    // --- Open / Close toggle ---
    var isOpen = false;
    function openWidget() {
      isOpen = true;
      iframe.style.display = "block";
      launcher.style.display = "none";
    }
    function closeWidget() {
      isOpen = false;
      iframe.style.display = "none";
      launcher.style.display = "flex";
    }
    launcher.addEventListener("click", openWidget);
    window.addEventListener("message", function (event) {
      if (event.data && event.data.type === "webai-widget-close") {
        closeWidget();
      }
    });

    // Auto-popup support
    var autoPopupDelay = { "1s": 1000, "2s": 2000, "3s": 3000, "5s": 5000 };
    var popupMs = autoPopupDelay[merged.autoPopup];
    if (popupMs) {
      setTimeout(function () { if (!isOpen) openWidget(); }, popupMs);
    }

    document.body.appendChild(launcher);
    document.body.appendChild(iframe);
  }

  var escalationUrl = apiBase + "/v1/pk/" + encodeURIComponent(botKey) + "/escalation-config";

  fetch(configUrl)
    .then(function (r) { return r.ok ? r.json() : {}; })
    .then(function (config) {
      return fetch(escalationUrl)
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (esc) {
          config = config || {};
          config.escalationsEnabled = true;
          return config;
        })
        .catch(function () { return config || {}; });
    })
    .then(function (config) { applyConfig(config || {}); })
    .catch(function () { applyConfig(scriptOverrides()); });
})();
