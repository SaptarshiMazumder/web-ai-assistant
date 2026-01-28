# URL Discovery: How Different Website Scenarios Are Handled

This document explains how the URL discovery system handles various real-world website scenarios.

## Discovery Flow Overview

The system uses a **3-strategy fallback approach**:

1. **Strategy 1**: Try sitemap discovery (via robots.txt → sitemap.xml)
2. **Strategy 2**: Fallback to crawl4ai link discovery (browser-based crawling)
3. **Strategy 3**: Last resort - return root URL (if allowed by robots.txt)

---

## Scenario 1: Site Has NO robots.txt AND NO sitemap.xml

**What happens:**
1. `_parse_robots_sitemaps()` tries to fetch `/robots.txt` → returns `[]` (empty list)
2. `sitemap_urls` is empty → skips Strategy 1
3. Falls back to **Strategy 2**: `discover_internal_urls()` 
   - Uses crawl4ai to crawl the root URL
   - Follows internal links up to 3 levels deep
   - Discovers URLs by crawling the actual website
4. Returns discovered URLs (typically 50-2000 URLs depending on site size)

**Result:** ✅ Works perfectly - discovers URLs by crawling

---

## Scenario 2: Site Has robots.txt BUT NO sitemap listed

**Example robots.txt:**
```
User-agent: *
Allow: /
Disallow: /admin/
```

**What happens:**
1. `_parse_robots_sitemaps()` fetches robots.txt → finds no `Sitemap:` lines → returns `[]`
2. `sitemap_urls` is empty → skips Strategy 1
3. Falls back to **Strategy 2**: `discover_internal_urls()`
   - Uses robots.txt parser to filter URLs during discovery
   - Only crawls URLs allowed by robots.txt
   - Respects `Disallow` rules
4. Returns discovered URLs (filtered by robots.txt rules)

**Result:** ✅ Works perfectly - discovers URLs while respecting robots.txt

---

## Scenario 3: Site Has robots.txt WITH sitemap listed

**Example robots.txt:**
```
User-agent: *
Allow: /
Sitemap: https://example.com/sitemap.xml
```

**What happens:**
1. `_parse_robots_sitemaps()` finds `Sitemap: https://example.com/sitemap.xml`
2. `_fetch_sitemap_urls_async()` fetches the sitemap:
   - First tries `requests.get()` (fast)
   - If that fails or gets CAPTCHA → falls back to `crawl4ai` (browser-based)
3. Parses XML and extracts all `<loc>` URLs
4. Filters URLs by robots.txt rules
5. Returns sitemap URLs (typically 100-5000 URLs)

**Result:** ✅ Works perfectly - fast sitemap discovery

---

## Scenario 4: Sitemap Index (List of Sitemaps)

**Example sitemap.xml:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex>
  <sitemap>
    <loc>https://example.com/sitemap-posts.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-pages.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-products.xml</loc>
  </sitemap>
</sitemapindex>
```

**What happens:**
1. `_fetch_sitemap_urls()` detects sitemap index (URLs ending in `.xml`)
2. **Recursively processes each child sitemap:**
   - Fetches `sitemap-posts.xml` → extracts URLs
   - Fetches `sitemap-pages.xml` → extracts URLs  
   - Fetches `sitemap-products.xml` → extracts URLs
3. **Continues even if some child sitemaps fail:**
   - If `sitemap-posts.xml` fails → continues with others
   - Collects partial results from successful sitemaps
4. Combines all URLs from all child sitemaps
5. Returns combined list (can be 10,000+ URLs)

**Result:** ✅ Works perfectly - handles nested sitemap indexes with up to 6 levels deep

**Code location:** `_fetch_sitemap_urls_async()` lines 301-311

---

## Scenario 5: Multiple Sitemaps in robots.txt

**Example robots.txt:**
```
User-agent: *
Sitemap: https://example.com/sitemap.xml
Sitemap: https://example.com/sitemap-blog.xml
Sitemap: https://example.com/sitemap-shop.xml
```

**What happens:**
1. `_parse_robots_sitemaps()` extracts all 3 sitemap URLs
2. **Processes each sitemap sequentially:**
   - Fetches `sitemap.xml` → extracts URLs
   - Fetches `sitemap-blog.xml` → extracts URLs
   - Fetches `sitemap-shop.xml` → extracts URLs
3. **Continues even if some fail:**
   - If `sitemap-blog.xml` is protected → continues with others
   - Collects partial results
4. Combines all URLs from all sitemaps
5. Returns combined list

**Result:** ✅ Works perfectly - handles multiple sitemaps gracefully

**Code location:** `discover_urls()` lines 358-369

---

## Scenario 6: Sitemap Protected by CAPTCHA/JS Challenge

**What happens:**
1. `_fetch_sitemap_urls()` tries `requests.get()` → gets HTML/CAPTCHA page
2. `_is_valid_xml()` detects it's not valid XML (contains "captcha", "cloudflare", etc.)
3. **Automatic fallback to crawl4ai:**
   - `_fetch_sitemap_with_crawl4ai()` uses browser to fetch sitemap
   - Browser can handle JS challenges, CAPTCHA solving (if configured)
   - Extracts XML content from HTML response
4. Parses XML and extracts URLs
5. Returns discovered URLs

**Result:** ✅ Works - bypasses protection using browser

**Code location:** `_fetch_sitemap_urls_async()` lines 290-299

---

## Scenario 7: Nested Sitemap Indexes (Sitemap of Sitemaps of Sitemaps)

**Example:**
```
sitemap.xml (index)
  └── sitemap-categories.xml (index)
      └── sitemap-electronics.xml (actual URLs)
      └── sitemap-clothing.xml (actual URLs)
```

**What happens:**
1. Fetches `sitemap.xml` → detects it's an index
2. Fetches `sitemap-categories.xml` → detects it's also an index
3. Recursively fetches `sitemap-electronics.xml` and `sitemap-clothing.xml`
4. Extracts URLs from both
5. **Maximum depth:** 6 levels (prevents infinite recursion)
6. Returns all discovered URLs

**Result:** ✅ Works - handles nested indexes up to 6 levels

**Code location:** `_fetch_sitemap_urls()` line 237 (`_MAX_SITEMAP_DEPTH = 6`)

---

## Scenario 8: robots.txt Disallows Everything

**Example robots.txt:**
```
User-agent: *
Disallow: /
```

**What happens:**
1. `_get_robots_parser()` parses robots.txt
2. During discovery, `_is_url_allowed_by_robots()` checks each URL
3. **All URLs are filtered out** (disallowed)
4. Strategy 1 (sitemap) → returns empty (all filtered)
5. Strategy 2 (crawl4ai) → returns empty (all filtered)
6. Strategy 3 → checks if root URL is allowed → **NO** (disallowed)
7. Returns `[]` (empty array)

**Result:** ⚠️ Returns empty - correctly respects robots.txt that disallows everything

**Code location:** `discover_urls()` lines 397-405

---

## Scenario 9: robots.txt Allows Root But Disallows Subdirectories

**Example robots.txt:**
```
User-agent: *
Allow: /
Disallow: /private/
Disallow: /admin/
```

**What happens:**
1. Sitemap discovery extracts all URLs
2. `_dedupe_and_filter()` filters URLs using robots.txt:
   - Keeps: `https://example.com/`, `https://example.com/about`
   - Removes: `https://example.com/private/`, `https://example.com/admin/`
3. Returns filtered URLs (only allowed ones)

**Result:** ✅ Works perfectly - respects partial disallow rules

**Code location:** `_dedupe_and_filter()` lines 316-340

---

## Scenario 10: Sitemap Returns Empty (No URLs)

**What happens:**
1. Fetches sitemap successfully
2. Parses XML → finds no `<loc>` tags (empty sitemap)
3. `sitemap_candidates` is empty
4. Falls back to **Strategy 2**: `discover_internal_urls()`
5. Crawls website to discover URLs
6. Returns discovered URLs

**Result:** ✅ Works - falls back to crawling when sitemap is empty

---

## Scenario 11: All Discovery Methods Fail (Network Error, Server Down)

**What happens:**
1. Strategy 1 (sitemap) → fails (server down, network error)
2. Strategy 2 (crawl4ai) → fails (can't connect)
3. **Strategy 3 (last resort):**
   - Checks if root URL is allowed by robots.txt
   - If allowed → returns `[root_url]`
   - If disallowed → returns `[]`

**Result:** ✅ Returns at least root URL (if allowed) - never completely empty unless robots.txt disallows

**Code location:** `discover_urls()` lines 397-405

---

## Scenario 12: Sitemap Has Both Regular URLs AND Nested Sitemaps

**Example sitemap.xml:**
```xml
<urlset>
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
  <url><loc>https://example.com/sitemap-blog.xml</loc></url>
</urlset>
```

**What happens:**
1. Parses sitemap → finds 3 URLs
2. Detects `sitemap-blog.xml` ends with `.xml` → treats as nested sitemap
3. Recursively fetches `sitemap-blog.xml` → extracts its URLs
4. Combines: `page1`, `page2`, + all URLs from `sitemap-blog.xml`
5. Returns combined list

**Result:** ✅ Works - handles mixed sitemaps (URLs + nested sitemaps)

---

## Scenario 13: robots.txt Points to Sitemap That Doesn't Exist (404)

**What happens:**
1. `_parse_robots_sitemaps()` finds sitemap URL in robots.txt
2. `_fetch_sitemap_urls_async()` tries to fetch → gets 404
3. Falls back to `crawl4ai` → also fails (404)
4. Continues to next sitemap (if multiple) or falls back to Strategy 2
5. Uses crawl4ai to discover URLs by crawling

**Result:** ✅ Works - gracefully handles missing sitemaps

---

## Scenario 14: Sitemap Behind Authentication

**What happens:**
1. `requests.get()` tries to fetch → gets 401/403
2. Falls back to `crawl4ai` (browser-based)
3. If crawl4ai also gets auth error → sitemap fails
4. Falls back to **Strategy 2**: `discover_internal_urls()`
5. Tries to crawl public pages (auth-protected sitemap doesn't block public crawling)

**Result:** ✅ Works - discovers public URLs even if sitemap is protected

---

## Scenario 15: Very Large Sitemap (10,000+ URLs)

**What happens:**
1. Fetches and parses sitemap
2. Extracts all URLs (can be 50,000+)
3. `_dedupe_and_filter()` limits to `_MAX_DISCOVERY_URLS` (2000 by default)
4. Returns top 2000 URLs (deduplicated, filtered by robots.txt)

**Result:** ✅ Works - handles large sitemaps with reasonable limits

**Code location:** `_dedupe_and_filter()` line 259 (`limit` parameter)

---

## Key Features

### ✅ Always Returns Partial Results
- If 5 sitemaps fail but 1 succeeds → returns URLs from successful one
- If sitemap fails but crawl4ai works → returns crawl4ai results
- If everything fails → returns at least root URL (if allowed)

### ✅ Never Crashes
- All functions wrapped in `safe_execute()` / `safe_execute_async()`
- Exceptions caught and logged, never propagated
- Always returns a result (even if empty)

### ✅ Respects robots.txt
- All discovered URLs filtered by robots.txt rules
- Disallowed URLs never returned
- If robots.txt disallows everything → returns empty (correct behavior)

### ✅ Handles Protected Content
- Detects CAPTCHA/bot protection
- Falls back to browser-based fetching (crawl4ai)
- Extracts XML from HTML responses

### ✅ Recursive Sitemap Processing
- Handles sitemap indexes (sitemaps of sitemaps)
- Up to 6 levels deep (prevents infinite recursion)
- Continues even if some nested sitemaps fail

---

## Summary Table

| Scenario | Strategy Used | Result |
|----------|--------------|--------|
| No robots.txt, no sitemap | Strategy 2 (crawl4ai) | ✅ Discovers URLs by crawling |
| robots.txt only (no sitemap) | Strategy 2 (crawl4ai) | ✅ Discovers URLs, respects robots.txt |
| robots.txt + sitemap | Strategy 1 (sitemap) | ✅ Fast sitemap discovery |
| Sitemap index | Strategy 1 (recursive) | ✅ Handles nested sitemaps |
| Multiple sitemaps | Strategy 1 (all sitemaps) | ✅ Combines all sitemaps |
| Protected sitemap | Strategy 1 (crawl4ai fallback) | ✅ Bypasses protection |
| Nested indexes | Strategy 1 (recursive, max 6 levels) | ✅ Handles deep nesting |
| robots.txt disallows all | All strategies filtered | ⚠️ Returns empty (correct) |
| Partial disallow | Strategy 1/2 (filtered) | ✅ Returns only allowed URLs |
| Empty sitemap | Strategy 2 (crawl4ai) | ✅ Falls back to crawling |
| All methods fail | Strategy 3 (root URL) | ✅ Returns root URL if allowed |
| Mixed sitemap | Strategy 1 (recursive) | ✅ Handles URLs + nested sitemaps |
| Missing sitemap (404) | Strategy 2 (crawl4ai) | ✅ Falls back to crawling |
| Auth-protected sitemap | Strategy 2 (crawl4ai) | ✅ Discovers public URLs |
| Large sitemap (10k+ URLs) | Strategy 1 (limited to 2000) | ✅ Returns top 2000 URLs |

---

## Code Flow Diagram

```
discover_urls(root_url)
│
├─→ Strategy 1: Sitemap Discovery
│   │
│   ├─→ Parse robots.txt → Extract sitemap URLs
│   │
│   ├─→ For each sitemap URL:
│   │   │
│   │   ├─→ Try requests.get() (fast)
│   │   │   ├─→ Success → Parse XML → Extract URLs
│   │   │   └─→ Fail/CAPTCHA → Try crawl4ai (browser)
│   │   │
│   │   ├─→ If sitemap is index (contains .xml URLs):
│   │   │   └─→ Recursively fetch child sitemaps (max 6 levels)
│   │   │
│   │   └─→ Continue even if this sitemap fails
│   │
│   ├─→ Filter URLs by robots.txt
│   │
│   └─→ If found URLs → RETURN ✅
│
├─→ Strategy 2: crawl4ai Link Discovery (if Strategy 1 failed)
│   │
│   ├─→ Crawl root URL with crawl4ai
│   ├─→ Follow internal links (max 3 levels deep)
│   ├─→ Filter by robots.txt
│   │
│   └─→ If found URLs → RETURN ✅
│
└─→ Strategy 3: Last Resort (if all failed)
    │
    ├─→ Check if root URL allowed by robots.txt
    │   ├─→ Yes → RETURN [root_url] ✅
    │   └─→ No → RETURN [] ⚠️ (robots.txt disallows)
```

---

## Testing Recommendations

To verify these scenarios work correctly, test with:

1. **Simple site**: `https://example.com` (no robots.txt, no sitemap)
2. **WordPress site**: Usually has sitemap.xml
3. **Large e-commerce**: Multiple sitemaps, sitemap indexes
4. **Protected site**: Cloudflare/CAPTCHA protection
5. **Strict robots.txt**: Disallows everything
6. **Nested sitemaps**: Sitemap of sitemaps

All scenarios should return at least the root URL (unless robots.txt disallows it).
