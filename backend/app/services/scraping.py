from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from markdownify import markdownify
from urllib.parse import urlparse, urljoin
import os
import hashlib

RAW_DIR = "raw_pages"
os.makedirs(RAW_DIR, exist_ok=True)

def safe_filename_from_url(url):
    domain = urlparse(url).netloc.replace(":", "_")
    url_hash = hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]
    return f"{domain}_{url_hash}.txt"

def save_raw_text(url, text):
    fname = safe_filename_from_url(url)
    fpath = os.path.join(RAW_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)

def scrape_with_sync_playwright(url, timeout=6000):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.route("**/*", lambda route, request: \
                route.abort() if request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_()
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            html = page.content()
            final_url = page.url
            browser.close()
            return html, final_url
    except Exception as e:
        print(f"[SCRAPE ERROR] Failed to fetch {url} -- {e}")
        return "", url

def extract_tables_as_markdown(soup):
    tables = []
    for table in soup.find_all("table"):
        md = []
        headers = []
        for th in table.find_all("th"):
            headers.append(th.get_text(strip=True))
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]
        if headers:
            md.append("| " + " | ".join(headers) + " |")
            md.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            md.append("| " + " | ".join(row) + " |")
        tables.append("\n".join(md))
    return tables

def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    noisy_selectors = [
        "nav", "footer", "aside", "form", "header",
        "[role=navigation]", "[role=contentinfo]", "[role=banner]", "[role=alert]", "[role=dialog]",
        "[class*=nav]", "[class*=footer]", "[class*=header]", "[class*=menu]", "[class*=sidebar]", "[class*=popup]", "[class*=cookie]",
        "[id*=nav]", "[id*=footer]", "[id*=header]", "[id*=menu]", "[id*=sidebar]", "[id*=popup]", "[id*=cookie]",
    ]
    for selector in noisy_selectors:
        for tag in soup.select(selector):
            tag.decompose()
    main = soup.find("main")
    if not main:
        main = soup.find(attrs={"role": "main"})
    if not main:
        main = soup.find("article")
    if not main:
        divs = soup.find_all("div")
        if divs:
            main = max(divs, key=lambda d: len(d.get_text()))
    if main:
        return str(main)
    return str(soup.body) if soup.body else str(soup)

def _extract_text_and_links(html: str, base_url: str):
    clean_html = extract_main_content(html)
    markdown = markdownify(clean_html, heading_style="ATX")
    soup = BeautifulSoup(html, "html.parser")
    page_links = [
        {
            "text": a.get_text(strip=True),
            "href": urljoin(base_url, a.get("href"))
        }
        for a in soup.find_all("a", href=True)
        if a.get("href", "").startswith(("http", "/"))
    ]
    page_links = [l for l in page_links if l["href"].startswith("http")]
    return markdown, page_links

def scrape_one(url: str):
    html, final_url = scrape_with_sync_playwright(url)
    text, links = _extract_text_and_links(html, final_url)
    save_raw_text(final_url, text)
    return {
        "url": final_url,
        "text": text,
        "links": links
    }