// siteCrawler.ts

export function debugLog(msg: string) {
  const div = document.getElementById("debug-log");
  if (div) {
    div.textContent += msg + "\n";
    div.scrollTop = div.scrollHeight;
  }
}

// Extract all same-domain links from HTML using raw href (never uses .href property)
export function extractSameDomainLinksFromHtml(html: string, baseUrl: string): { text: string; href: string }[] {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, "text/html");
  const origin = new URL(baseUrl).origin;

  debugLog(`Extracting links from: ${origin}`);

  // Print all raw hrefs before filtering
  const allLinks = Array.from(doc.querySelectorAll("a"));
  debugLog(`All <a> links found (${allLinks.length}):`);
  allLinks.forEach(a => {
    const rawHref = a.getAttribute("href") || "";
    debugLog(`  - ${rawHref} (${a.innerText.trim()})`);
  });

  let links: { text: string; href: string }[] = [];
  allLinks.forEach(a => {
    const rawHref = a.getAttribute("href") || "";
    const outer = a.outerHTML;
    if (
      outer.includes(origin) && // filter on raw HTML if you want
      rawHref &&
      !rawHref.endsWith("#") &&
      !rawHref.startsWith("javascript:") &&
      rawHref !== baseUrl
    ) {
      // Always resolve as absolute for crawling
      try {
        const absHref = new URL(rawHref, baseUrl).href;
        links.push({ text: a.innerText.trim(), href: absHref });
      } catch {}
    }
  });

  // Remove duplicates by resolved href
  const uniqueLinksMap = new Map<string, { text: string; href: string }>();
  links.forEach(link => {
    if (!uniqueLinksMap.has(link.href)) {
      uniqueLinksMap.set(link.href, link);
    }
  });

  debugLog(`Same-domain links found (${uniqueLinksMap.size}):`);
  Array.from(uniqueLinksMap.values()).forEach(link => {
    debugLog(`  - ${link.href} (${link.text})`);
  });

  return Array.from(uniqueLinksMap.values());
}

export async function crawlEntireSite(startUrl: string, domain: string, backendUrl: string) {
  const visited = new Set<string>();
  const queue = [startUrl];

  while (queue.length > 0) {
    const url = queue.shift()!;
    if (visited.has(url)) continue;
    visited.add(url);

    debugLog(`Crawling: ${url}`);

    try {
      const resp = await fetch(url);
      const html = await resp.text();

      debugLog(`Fetched (${html.length} chars): ${url}`);

      await fetch(`${backendUrl}/add_page_data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, html, domain }),
      });

      const links = extractSameDomainLinksFromHtml(html, url);
      debugLog(`Found ${links.length} same-domain links on ${url}:`);
      links.forEach(link => debugLog(`  - ${link.href} (${link.text})`));
      for (const link of links) {
        if (!visited.has(link.href)) queue.push(link.href);
      }

    } catch (err) {
      debugLog("Error crawling " + url + ": " + err);
    }
  }
}
