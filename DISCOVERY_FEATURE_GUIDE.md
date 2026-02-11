# URL Discovery Feature - Quick Guide

## What Was Implemented

I've added **URL discovery and path-based crawling** to the "shared provider" route in your create bot flow. This allows users to automatically discover and add pages from a specific section of a website.

## Key Features

### ✅ Path-Based Discovery
- Enter a URL like `https://beauty.hotpepper.jp/slnH000447147/`
- Discovers **ONLY** pages within that path
- **Does NOT** crawl the entire domain or external links

### ✅ Real-Time Streaming
- URLs appear as they're discovered
- Live progress counter
- Can be stopped at any time

### ✅ Smart Selection
- All URLs auto-selected by default
- Bulk select/deselect all
- Individual URL selection

### ✅ Auto-Generated Labels
- Automatically creates labels from URL paths
- Example: `https://example.com/pricing/` → Label: "Pricing"

## How to Use

1. **Navigate to shared provider page**
   - Go to "Add your page links" in create bot flow
   - (This is for "shared" hosting type)

2. **Click "Discover pages from URL"**
   - Discovery panel appears

3. **Enter a URL with specific path**
   ```
   https://beauty.hotpepper.jp/slnH000447147/
   ```

4. **Click "Discover" or press Enter**
   - Discovery begins streaming URLs
   - Progress shows: "Discovering pages... 15 found so far"

5. **Review discovered URLs**
   - All URLs are pre-selected
   - Select/deselect as needed
   - Use "Select all" / "Deselect all" for bulk operations

6. **Click "Add X pages"**
   - Selected URLs are added to your page links
   - Labels are auto-generated
   - You can edit labels before training

## Examples

### Example 1: Salon on Marketplace

**Input:**
```
https://beauty.hotpepper.jp/slnH000447147/
```

**Discovers:**
- ✅ `https://beauty.hotpepper.jp/slnH000447147/stylist/`
- ✅ `https://beauty.hotpepper.jp/slnH000447147/pricing/`
- ✅ `https://beauty.hotpepper.jp/slnH000447147/style/L255896009.html`

**Does NOT Discover:**
- ❌ `https://beauty.hotpepper.jp/other-salon/` (different path)
- ❌ `https://example.com/` (external domain)

### Example 2: Hotel Section

**Input:**
```
https://hotelchain.com/properties/tokyo-shiodome/
```

**Discovers:**
- ✅ `https://hotelchain.com/properties/tokyo-shiodome/rooms/`
- ✅ `https://hotelchain.com/properties/tokyo-shiodome/dining/`
- ✅ `https://hotelchain.com/properties/tokyo-shiodome/booking/`

**Does NOT Discover:**
- ❌ `https://hotelchain.com/properties/osaka/` (different hotel)
- ❌ `https://hotelchain.com/about/` (different path)

## Technical Implementation

### Frontend Changes
- **File Modified:** `dashboard/src/pages/createBot/CreateBotSharedUrlsPage.tsx`
- **Lines Added:** ~320 lines
- **New Features:**
  - Discovery toggle button
  - URL input with validation
  - Real-time streaming display
  - URL selection interface
  - Smart label generation

### Backend Changes
- **None!** The backend already supports path-based filtering through:
  - `/v1/org/url-discovery/stream` endpoint
  - `_is_url_under_root_path()` function in `crawl_service.py`

### How Path-Based Filtering Works

```python
def _is_url_under_root_path(url: str, root_url: str) -> bool:
    """
    Returns True if:
    1. URL has same domain as root_url
    2. URL path is under root_url's path
    
    Example:
    root: https://example.com/hotel/tokyo/
    ✅ https://example.com/hotel/tokyo/rooms/
    ✅ https://example.com/hotel/tokyo/dining/index.html
    ❌ https://example.com/hotel/osaka/
    ❌ https://example.com/
    """
```

## UI Components

### Discovery Panel
```
┌─────────────────────────────────────────────────────────┐
│ Discover pages                                       [×] │
├─────────────────────────────────────────────────────────┤
│ Enter a URL to discover all pages within that path.    │
│ For example, https://example.com/hotel/tokyo/ will     │
│ find all pages under that hotel section.               │
│                                                         │
│ ┌────────────────────────────────────────┐             │
│ │ https://example.com/section/            │ [Discover] │
│ └────────────────────────────────────────┘             │
│                                                         │
│ 🔵🔵🔵 Discovering pages... 15 found so far            │
│                                                         │
│ Found 15 pages                            [Select all] │
│ ┌────────────────────────────────────────────────────┐ │
│ │ ☑ https://example.com/section/pricing/            │ │
│ │ ☑ https://example.com/section/services/           │ │
│ │ ☑ https://example.com/section/contact/            │ │
│ │ ...                                                │ │
│ └────────────────────────────────────────────────────┘ │
│                                                         │
│                              [Cancel]  [Add 15 pages]   │
└─────────────────────────────────────────────────────────┘
```

## Benefits

### For Users
1. **Save time** - No manual URL entry for each page
2. **Complete coverage** - Automatically finds all pages in a section
3. **Accurate** - Only discovers relevant pages within the path
4. **Flexible** - Can select/deselect discovered URLs

### For Bot Quality
1. **Comprehensive knowledge** - Discovers pages users might miss
2. **Focused training** - Only trains on relevant section
3. **Organized** - Auto-generated labels help categorize content

## Testing Checklist

- [x] URL validation (invalid URLs show error)
- [x] Path-based filtering (only discovers URLs within path)
- [x] Real-time streaming (URLs appear as discovered)
- [x] Stop discovery (abort controller works)
- [x] Select/deselect URLs (individual and bulk)
- [x] Add discovered URLs (no duplicates)
- [x] Label generation (extracts from URL path)
- [x] Integration with existing rows (preserves manual URLs)
- [x] No linter errors
- [x] Builds successfully in Docker

## Files Modified

```
dashboard/src/pages/createBot/CreateBotSharedUrlsPage.tsx
```

## Documentation Created

```
SHARED_PROVIDER_URL_DISCOVERY.md
DISCOVERY_FEATURE_GUIDE.md
```

## Next Steps

1. **Test the feature** - Try discovering URLs from different websites
2. **User feedback** - Get feedback on UX and functionality
3. **Iterate** - Add enhancements based on user needs

## Support

If you encounter any issues:
1. Check browser console for errors
2. Verify the URL is accessible
3. Check that the path contains discoverable pages
4. Try stopping and restarting discovery

---

**Feature is ready for testing!** 🎉
