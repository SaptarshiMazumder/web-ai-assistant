declare global {
  interface Window {
    find(
      string: string,
      caseSensitive?: boolean,
      backwards?: boolean,
      wrap?: boolean,
      wholeWord?: boolean,
      searchInFrames?: boolean,
      showDialog?: boolean
    ): boolean;
  }
}

function extractTablesAsMarkdown(): string[] {
  const tables: string[] = [];
  document.querySelectorAll("table").forEach((table) => {
    const rows = Array.from(table.rows);
    if (rows.length === 0) return;
    const headers = Array.from(rows[0].cells).map((cell) => cell.textContent?.trim() ?? "");
    const headerLine = "| " + headers.join(" | ") + " |";
    const divider = "| " + headers.map(() => "---").join(" | ") + " |";
    const bodyLines = rows.slice(1).map((row) => {
      const cells = Array.from(row.cells).map((cell) => cell.textContent?.trim() ?? "");
      return "| " + cells.join(" | ") + " |";
    });
    tables.push([headerLine, divider, ...bodyLines].join("\n"));
  });
  return tables;
}

// --- Utility: get all same-domain links (unique, clean) ---
function extractSameDomainLinks(): { text: string; href: string }[] {
  const origin = location.origin;
  const links = Array.from(document.querySelectorAll("a"))
    .filter((a) =>
      a.href &&
      a.href.startsWith(origin) &&  // Only same-origin
      !a.href.endsWith("#") &&
      !a.href.startsWith("javascript:") &&
      a.href !== location.href
    )
    .map((a) => ({
      text: a.innerText.trim(),
      href: a.href,
    }));

  // Remove duplicates by href
  const uniqueLinksMap = new Map<string, { text: string; href: string }>();
  links.forEach(link => {
    if (!uniqueLinksMap.has(link.href)) {
      uniqueLinksMap.set(link.href, link);
    }
  });
  return Array.from(uniqueLinksMap.values());
}


chrome.runtime.onMessage.addListener((req, sender, sendResp) => {
  
   if (req.type === "PING") {
    sendResp({ pong: true });
  }
  if (req.type === "GET_PAGE_DATA") {
    // 1. Visible text
    const text = document.body.innerText;

    // 2. Tables as markdown
    const tables = extractTablesAsMarkdown();

    // 3. All links: text and href
    const links = Array.from(document.querySelectorAll("a"))
      .filter((a) => a.href && a.innerText.trim().length > 0)
      .map((a) => ({
        text: a.innerText.trim(),
        href: a.href,
      }));

    // 4. Images: alt and src
    const images = Array.from(document.querySelectorAll("img")).map((img) => ({
      alt: img.alt,
      src: img.src,
    }));

    sendResp?.({ text, tables, links, images });
  }

  // --- NEW: Give all same-domain links for "site QA" ---
  if (req.type === "GET_ALL_SAME_DOMAIN_LINKS") {
    const pageLinks = extractSameDomainLinks();
    sendResp?.({ links: pageLinks });
  }

  if (req.type === "GET_PAGE_TEXT") {
    sendResp?.({ text: document.body.innerText });
  }

  if (req.type === "JUMP_TO_POSITION") {
    const excerpt = req.excerpt || "";
    if (!excerpt) return;

    let found = false;
    let toTry = [
      excerpt,
      excerpt.trim(),
      excerpt.slice(0, 60),
      excerpt.split(" ").slice(0, 6).join(" "),
      excerpt.split(" ").slice(0, 10).join(" "),
    ];

    for (const snippet of toTry) {
      if (!snippet) continue;
      if (window.find(snippet, false, false, true, false, false, false)) {
        found = true;
        break;
      }
    }

    if (!found) {
      alert("Could not locate the answer in the page.");
    }
  }
});

export {};
