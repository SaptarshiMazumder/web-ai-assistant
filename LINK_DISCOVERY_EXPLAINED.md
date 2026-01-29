# How Link Discovery and Visit Tracking Works

## Overview

The system uses a **Breadth-First Search (BFS)** algorithm to discover URLs by crawling pages level by level, extracting links from each page, and following those links to discover more pages.

---

## Data Structures

### 1. **`visited` Set** (Tracks Crawled URLs)
```python
visited = set()  # Example: {"https://example.com/", "https://example.com/articles", ...}
```
- **Purpose**: Prevents crawling the same URL twice
- **When added**: Immediately when a URL is crawled (line 499)
- **Normalization**: URLs are normalized (removes fragments like `#section`) before adding

### 2. **`discovered` List** (Tracks All Found URLs)
```python
discovered = []  # Example: ["https://example.com/", "https://example.com/articles", ...]
```
- **Purpose**: Stores all unique URLs found during discovery
- **When added**: When a URL is successfully crawled and is internal (line 501)
- **Final result**: This is what gets returned to the user

### 3. **`current_urls` Set** (URLs to Crawl at Current Depth)
```python
current_urls = set([root_url])  # Starts with root URL
```
- **Purpose**: URLs queued for crawling at the current depth level
- **Updated**: After each depth iteration, becomes `next_level_urls`

### 4. **`next_level_urls` Set** (URLs Found for Next Depth)
```python
next_level_urls = set()  # Links extracted from current pages
```
- **Purpose**: Collects links found on current pages to crawl in next iteration
- **Becomes**: `current_urls` for next depth level

---

## The Discovery Algorithm (BFS)

### Step-by-Step Process:

#### **Initialization:**
```python
visited = set()                    # Empty - nothing crawled yet
current_urls = {root_url}          # Start with root URL
discovered = []                    # Empty - nothing discovered yet
```

#### **For Each Depth Level (0 to max_depth-1):**

**1. Filter URLs to Crawl:**
```python
urls_to_crawl = [u for u in current_urls if u not in visited]
```
- Only crawl URLs we haven't visited yet
- Example: If `current_urls = {"/articles", "/about"}` and `visited = {"/articles"}`, then `urls_to_crawl = ["/about"]`

**2. Batch Crawl Pages:**
```python
results = await crawler.arun_many(urls=urls_to_crawl, ...)
```
- Uses crawl4ai to fetch multiple pages simultaneously (up to `max_concurrent=20`)
- Returns results for each URL (success or failure)

**3. Process Each Result:**

**a) Mark URL as Visited:**
```python
norm = _normalize_url(result.url)  # Remove #fragment
visited.add(norm)                  # Mark as crawled
```
- **Why normalize?** `https://example.com/page#section` and `https://example.com/page` are the same page
- **Prevents**: Crawling the same page multiple times

**b) Add to Discovered List:**
```python
if norm not in discovered and is_internal(norm):
    discovered.append(norm)
```
- Only add if:
  - Not already in discovered list
  - Is internal (same domain as root URL)
- **Example**: `https://example.com/articles` ✅ vs `https://other.com/page` ❌

**c) Extract Links from Page:**
```python
links = result.links  # crawl4ai extracts this from HTML
```
- **What crawl4ai extracts**:
  - All `<a href="...">` tags from HTML
  - Links from JavaScript-rendered content (after JS execution)
  - Links from both static HTML and dynamic content
- **Structure**: `{"internal": [...], "external": [...]}`

**d) Process Internal Links:**
```python
for link in links.get("internal", []):
    href = _normalize_url(link.get("href"))
    if href not in visited and is_internal(href):
        if href not in discovered:
            next_level_urls.add(href)
```
- **Checks**:
  1. URL is normalized (no fragments)
  2. Not already visited (won't crawl again)
  3. Is internal (same domain)
  4. Not already discovered (avoid duplicates)
- **Adds to**: `next_level_urls` for next depth level

**4. Move to Next Depth:**
```python
current_urls = next_level_urls  # URLs found become next to crawl
next_level_urls = set()        # Reset for next iteration
```

---

## Example Walkthrough

**Starting URL:** `https://example.com/`

### **Depth 0 (Root Level):**
```
visited = {}
current_urls = {"https://example.com/"}
discovered = []

1. Crawl: https://example.com/
   - visited.add("https://example.com/")
   - discovered.append("https://example.com/")
   - Extract links: ["/articles", "/about", "/contact"]
   - next_level_urls = {"/articles", "/about", "/contact"}

Result:
  visited = {"https://example.com/"}
  discovered = ["https://example.com/"]
  current_urls = {"/articles", "/about", "/contact"}  (for next depth)
```

### **Depth 1:**
```
visited = {"https://example.com/"}
current_urls = {"/articles", "/about", "/contact"}
discovered = ["https://example.com/"]

1. Crawl: /articles
   - visited.add("/articles")
   - discovered.append("/articles")
   - Extract links: ["/articles/nutrition", "/articles/fitness"]
   - next_level_urls.add("/articles/nutrition")
   - next_level_urls.add("/articles/fitness")

2. Crawl: /about
   - visited.add("/about")
   - discovered.append("/about")
   - Extract links: []  (no links)
   - (nothing added to next_level_urls)

3. Crawl: /contact
   - visited.add("/contact")
   - discovered.append("/contact")
   - Extract links: ["/contact/form"]
   - next_level_urls.add("/contact/form")

Result:
  visited = {"https://example.com/", "/articles", "/about", "/contact"}
  discovered = ["https://example.com/", "/articles", "/about", "/contact"]
  current_urls = {"/articles/nutrition", "/articles/fitness", "/contact/form"}
```

### **Depth 2:**
```
visited = {"https://example.com/", "/articles", "/about", "/contact"}
current_urls = {"/articles/nutrition", "/articles/fitness", "/contact/form"}
discovered = ["https://example.com/", "/articles", "/about", "/contact"]

1. Crawl: /articles/nutrition
   - visited.add("/articles/nutrition")
   - discovered.append("/articles/nutrition")
   - Extract links: ["/articles/nutrition/bmr"]
   - next_level_urls.add("/articles/nutrition/bmr")

2. Crawl: /articles/fitness
   - visited.add("/articles/fitness")
   - discovered.append("/articles/fitness")
   - Extract links: []
   - (nothing added)

3. Crawl: /contact/form
   - visited.add("/contact/form")
   - discovered.append("/contact/form")
   - Extract links: []
   - (nothing added)

Result:
  visited = {"https://example.com/", "/articles", "/about", "/contact", 
             "/articles/nutrition", "/articles/fitness", "/contact/form"}
  discovered = ["https://example.com/", "/articles", "/about", "/contact",
                 "/articles/nutrition", "/articles/fitness", "/contact/form"]
  current_urls = {"/articles/nutrition/bmr"}
```

### **Depth 3:**
```
visited = {..., "/articles/nutrition", "/articles/fitness", "/contact/form"}
current_urls = {"/articles/nutrition/bmr"}
discovered = [..., "/articles/nutrition", "/articles/fitness", "/contact/form"]

1. Crawl: /articles/nutrition/bmr
   - visited.add("/articles/nutrition/bmr")
   - discovered.append("/articles/nutrition/bmr")
   - Extract links: []  (no more links)
   - next_level_urls = {}  (empty)

Result:
  visited = {..., "/articles/nutrition/bmr"}
  discovered = [..., "/articles/nutrition/bmr"]
  current_urls = {}  (empty - no more URLs to crawl)

Next iteration: urls_to_crawl = [] → breaks early ✅
```

---

## Key Logic Points

### **1. URL Normalization:**
```python
def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]  # Removes #fragment
```
- `https://example.com/page#section` → `https://example.com/page`
- Prevents treating same page as different URLs

### **2. Internal Link Check:**
```python
def is_internal(url: str) -> bool:
    return urlparse(url).netloc == root_netloc
```
- `https://example.com/articles` ✅ (same domain)
- `https://other.com/page` ❌ (different domain)
- `mailto:email@example.com` ❌ (not HTTP/HTTPS)

### **3. Visit Tracking:**
- **`visited`**: URLs that have been **crawled** (HTTP request made)
- **`discovered`**: URLs that have been **found** (may or may not be crawled yet)
- **Why both?** 
  - `visited` prevents re-crawling
  - `discovered` tracks all found URLs (some might be queued for future crawling)

### **4. Link Extraction Sources:**
- **Primary**: `result.links.get("internal", [])` - Links crawl4ai identifies as internal
- **Secondary**: `result.links.get("external", [])` - Check if "external" links are actually same domain
- **Why?** Some sites incorrectly mark internal links as external

### **5. BFS vs DFS:**
- **BFS (Breadth-First)**: Crawls all pages at depth 1, then all at depth 2, etc.
  - ✅ Finds pages closer to root first
  - ✅ More balanced discovery
  - ✅ Better for finding important pages early
- **DFS (Depth-First)**: Would go deep into one branch before exploring others
  - ❌ Might miss important pages in other branches
  - ❌ Less balanced

---

## What Happens When a Link is Found Multiple Times?

**Example:**
- Page A links to `/articles`
- Page B also links to `/articles`
- Page C also links to `/articles`

**Process:**
1. **First time** (from Page A):
   - `visited` doesn't contain `/articles` → will crawl it
   - `discovered` doesn't contain `/articles` → add to discovered
   - Add to `next_level_urls`

2. **Second time** (from Page B):
   - `visited` doesn't contain `/articles` yet (not crawled) → will still crawl
   - `discovered` contains `/articles` → skip adding to discovered
   - Skip adding to `next_level_urls` (already there or will be crawled)

3. **After crawling** `/articles`:
   - `visited.add("/articles")` → marked as crawled
   - Future finds of `/articles` → `if href not in visited` → **False** → skipped

**Result:** Each URL is crawled **exactly once**, even if found on multiple pages.

---

## Edge Cases Handled

### **1. Circular Links:**
```
Page A → Page B → Page A
```
- **Solution**: `visited` set prevents infinite loops
- Page A crawled → Page B found → Page B crawled → Page A found again → skipped (already in `visited`)

### **2. Duplicate Links on Same Page:**
```
Page has: <a href="/articles">Link 1</a> and <a href="/articles">Link 2</a>
```
- **Solution**: `next_level_urls` is a `set()` → automatically deduplicates
- Only one `/articles` added to queue

### **3. Links to Already Discovered URLs:**
```
Depth 1: Found /articles
Depth 2: Found /articles/nutrition which links back to /articles
```
- **Solution**: `if href not in discovered` check
- `/articles` already discovered → skip adding to `next_level_urls`
- But if not yet crawled, it will be crawled when its turn comes

### **4. External Links Marked as Internal:**
```
Link: https://example.com/page (marked as "external" by crawl4ai)
```
- **Solution**: Check both `internal` and `external` link lists
- Verify with `is_internal()` function
- Add if actually same domain

### **5. Failed Page Crawls:**
```
Page /articles fails to load (404, timeout, etc.)
```
- **Solution**: Still try to extract links from failed result
- `result.links` might still contain links even if page failed
- Continue to next page instead of stopping

---

## Summary

**Link Discovery:**
1. crawl4ai extracts all `<a href>` links from HTML (after JS execution)
2. Links are normalized (remove fragments)
3. Only internal links (same domain) are kept
4. Duplicates are filtered out using sets

**Visit Tracking:**
1. `visited` set tracks crawled URLs (prevents re-crawling)
2. `discovered` list tracks all found URLs (final result)
3. URLs are normalized before checking/adding
4. Each URL is crawled exactly once

**BFS Algorithm:**
1. Start with root URL
2. For each depth level:
   - Crawl all URLs at current level
   - Extract links from each page
   - Add found links to next level queue
3. Repeat until max_depth reached or no more links found

This ensures comprehensive discovery while avoiding infinite loops and duplicate work.
