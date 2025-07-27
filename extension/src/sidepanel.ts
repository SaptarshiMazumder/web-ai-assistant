import { marked } from "marked";
import hljs from "highlight.js";

const chatDiv = document.getElementById("chat")!;
const askBtn = document.getElementById("ask-btn")!;

askBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="#fff"/></svg>`;
askBtn.title = "Send";
const questionInput = document.getElementById("question")! as HTMLInputElement;

// --- DROPDOWN to toggle between Ask and Ask Smart ---

const dropdown = document.createElement("select");
dropdown.id = "ask-mode";
dropdown.style.marginLeft = "8px";

const optionAsk = document.createElement("option");
optionAsk.value = "ask";
optionAsk.textContent = "Ask";
dropdown.appendChild(optionAsk);

const optionSmart = document.createElement("option");
optionSmart.value = "smart";
optionSmart.textContent = "Ask Smart";
dropdown.appendChild(optionSmart);

// --- Smart Ask button (created but hidden by default) ---
const smartBtn = document.createElement("button");
smartBtn.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="#fff"/></svg>`;
smartBtn.title = "Send";
smartBtn.id = "smart-ask-btn";
smartBtn.style.marginLeft = "8px";
smartBtn.style.display = "none";

const inputRow = document.getElementById("inputRow");
// askBtn is already there after input, so insert smartBtn after askBtn
inputRow?.insertBefore(smartBtn, askBtn.nextSibling);
// dropdown always after smartBtn
inputRow?.insertBefore(dropdown, smartBtn.nextSibling);


// -- REMOVE the "site QA" button section completely --

// Utility: get all same-domain links from the active tab
function getSameDomainLinksFromActiveTab(): Promise<string[]> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]?.id) {
        resolve([]);
        return;
      }
      chrome.tabs.sendMessage(
        tabs[0].id,
        { type: "GET_ALL_SAME_DOMAIN_LINKS" },
        (resp) => {
          if (chrome.runtime.lastError || !resp) {
            // Content script not loaded or tab not accessible
            resolve([]);
            return;
          }
          resolve(resp.links || []);
        }
      );
    });
  });
}

// Existing: get current page data
function getPageDataFromActiveTab(): Promise<{
  text: string;
  tables: string[];
  links: Array<{ text: string; href: string }>;
  images: Array<{ alt: string; src: string }>;
}> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id !== undefined) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { type: "GET_PAGE_DATA" },
          (resp) => {
            resolve(resp || { text: "", tables: [], links: [], images: [] });
          }
        );
      } else {
        resolve({ text: "", tables: [], links: [], images: [] });
      }
    });
  });
}

// Markdown rendering for chat bubbles (fix: no async needed)
function appendMessage(text: string, sender: 'user' | 'bot' | 'thinking'): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  if (sender === 'bot') {
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

function jumpToSource(excerpt: string) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.id !== undefined) {
      chrome.tabs.sendMessage(
        tabs[0].id,
        { type: "JUMP_TO_POSITION", excerpt },
        () => {}
      );
    }
  });
}

// --- Standard "current page" ask ---
askBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage(question, "user");
  const thinkingBubble = appendMessage("Thinking...", "thinking");

  const pageData = await getPageDataFromActiveTab();
  let context = pageData.text;
  if (pageData.tables && pageData.tables.length > 0) {
    context += "\n\n# Tables:\n";
    context += pageData.tables.map((t, i) => `Table ${i+1}:\n${t}`).join("\n\n");
  }
  if (pageData.links && pageData.links.length > 0) {
    context += "\n\n# Links:\n";
    context += pageData.links.map((l, i) => `- [${l.text}](${l.href})`).join("\n");
  }
  if (pageData.images && pageData.images.length > 0) {
    context += "\n\n# Images:\n";
    context += pageData.images.map((img, i) => `- alt: "${img.alt}" src: ${img.src}`).join("\n");
  }

  try {
    const resp = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: context, question }),
    });
    const data = await resp.json();
    thinkingBubble.remove();
    appendMessage(data.answer, "bot");
    if (data.sources && data.sources.length > 0) {
      renderSources(data.sources);
    }
  } catch (err) {
    thinkingBubble.textContent = "Error. Please try again.";
  }
  questionInput.value = "";
};

// --- Smart Ask button logic ---
smartBtn.onclick = async function () {
  const question = questionInput.value.trim();
  if (!question) return;

  // Add "Smart QA" tag above the bubble
  const tag = document.createElement('div');
  tag.textContent = "Smart QA";
  tag.style.fontSize = "0.8em";
  tag.style.color = "#0a5";
  tag.style.fontWeight = "bold";
  tag.style.letterSpacing = "0.03em";
  tag.style.margin = "0 10px 0 0";
  tag.style.textAlign = "right";
  tag.style.width = "100%";
  chatDiv.appendChild(tag);
  appendMessage(question, "user");
  const thinkingBubble = appendMessage("Thinking (smart)...", "thinking");

  // Get tab/page data as before (get text, links, url)
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const tab = tabs[0];
    const page_url = tab?.url ?? "";
    chrome.tabs.sendMessage(
      tab.id!,
      { type: "GET_PAGE_DATA" },
      async (pageData) => {
        const body = {
          text: pageData.text,
          question,
          links: pageData.links,
          page_url,
        };
        try {
          const resp = await fetch("http://localhost:5000/ask-smart", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          const data = await resp.json();
          thinkingBubble.remove();

          // Show the answer
          appendMessage(data.answer, "bot");
          // Show sources (if any)
          if (data.sources && data.sources.length > 0) {
            renderSources(data.sources);
          }

          // If not sufficient, show LLM-picked links (plain, no style)
          if (data.sufficient === false && data.selected_links && data.selected_links.length > 0) {
            appendMessage("Try checking one of these links for more info:", "bot");
            data.selected_links.forEach((l: any) => {
              appendMessage(`• ${l.text} — ${l.href}`, "bot");
            });
          }

          if (data.visited_urls && data.visited_urls.length > 0) {
            const urls = data.visited_urls;
            let msg = "Pages visited:\n";
            urls.forEach((url: string, idx: number) => {
              msg += `• ${url}\n`;
            });
            // Subtle suggestion for last visited page:
            msg += `\nYou can also view the last visited page for more details..`;
            appendMessage(msg, "bot");
          }


        } catch (err) {
          thinkingBubble.textContent = "Error (smart QA). Please try again.";
        }
        questionInput.value = "";
      }
    );
  });
};

// --- DROPDOWN LOGIC: toggle visible button ---
dropdown.addEventListener("change", () => {
  if (dropdown.value === "ask") {
    askBtn.style.display = "inline-block";
    smartBtn.style.display = "none";
  } else {
    askBtn.style.display = "none";
    smartBtn.style.display = "inline-block";
  }
});

// Default to "Ask"
dropdown.value = "ask";
askBtn.style.display = "inline-block";
smartBtn.style.display = "none";
