# Shared Provider URL Discovery Implementation

## Overview

This document describes the implementation of URL discovery and crawling for the "shared provider" route in the create bot flow. This feature allows users to discover all pages within a specific URL path and automatically add them to their bot's knowledge base.

## Feature Description

When users select the "shared provider" hosting option, they can now:

1. **Enter a base URL** with a specific path (e.g., `https://beauty.hotpepper.jp/slnH000447147/`)
2. **Discover all pages** within that path prefix automatically
3. **Select which discovered pages** to add to their bot
4. **Automatically generate labels** for each discovered page based on the URL structure

## How It Works

### Path-Based Filtering

The system uses the existing `_is_url_under_root_path` function in the backend to ensure that only URLs within the specified path are discovered. For example:

- **Input URL**: `https://beauty.hotpepper.jp/slnH000447147/`
- **Discovers**: 
  - ✅ `https://beauty.hotpepper.jp/slnH000447147/staff/`
  - ✅ `https://beauty.hotpepper.jp/slnH000447147/pricing/`
  - ✅ `https://beauty.hotpepper.jp/slnH000447147/style/L255896009.html`
- **Does NOT discover**:
  - ❌ `https://beauty.hotpepper.jp/other-salon/`
  - ❌ `https://example.com/external-link/`

### Backend Implementation

The backend already supports path-based filtering through:

```python
def _is_url_under_root_path(url: str, root_url: str) -> bool:
    """
    True if url has same domain as root_url AND path is under root_url's path.
    E.g. root https://example.com/hotel/tokyoshiodome/ → allows /hotel/tokyoshiodome/* only.
    If root path is / (domain root), allows all paths on that domain.
    """
```

This function is used throughout the discovery adapters (`HttpUrlDiscoveryAdapter`, `Crawl4AIUrlDiscoveryAdapter`, `AutoUrlDiscoveryAdapter`) to filter discovered URLs.

**No backend changes were needed** - the existing discovery endpoint at `/v1/org/url-discovery/stream` already supports this functionality.

### Frontend Implementation

#### File Modified
- `dashboard/src/pages/createBot/CreateBotSharedUrlsPage.tsx`

#### Key Features

1. **Discovery Toggle Button**
   - Users can click "Discover pages from URL" to show the discovery panel
   - The panel can be closed at any time

2. **URL Input with Validation**
   - Users enter a URL (e.g., `https://example.com/section/`)
   - The URL is normalized (adds `https://` if missing)
   - Validation ensures the URL is valid before discovery starts

3. **Real-Time Discovery Stream**
   - Uses the existing `discoverUrls` hook from `useDashboardData`
   - Shows live progress: "Discovering pages... 15 found so far"
   - Animated loading dots using the existing `discovery-dot` CSS animation

4. **URL Selection**
   - All discovered URLs are auto-selected by default
   - Users can select/deselect individual URLs
   - "Select all" / "Deselect all" buttons for bulk operations
   - Scrollable list (max height 300px) for many URLs

5. **Smart Label Generation**
   - Automatically generates labels from URL paths
   - Example: `https://example.com/hotel/pricing/` → Label: "Pricing"
   - Converts dashes/underscores to spaces and capitalizes words

6. **Integration with Existing Rows**
   - Discovered URLs are added to the existing shared URL rows
   - Duplicates are automatically prevented
   - Existing manually-entered URLs are preserved

#### State Management

```typescript
const [showDiscovery, setShowDiscovery] = useState(false)
const [discoveryUrl, setDiscoveryUrl] = useState('')
const [discoveredUrls, setDiscoveredUrls] = useState<string[]>([])
const [selectedDiscoveredUrls, setSelectedDiscoveredUrls] = useState<Set<string>>(new Set())
const [isDiscovering, setIsDiscovering] = useState(false)
const [discoveryError, setDiscoveryError] = useState<string | null>(null)
const discoveryAbortRef = useRef<AbortController | null>(null)
```

#### Key Functions

- `handleDiscoverUrls()`: Initiates URL discovery
- `handleStopDiscovery()`: Aborts the discovery process
- `handleAddDiscoveredUrls()`: Adds selected URLs to the shared URL rows
- `toggleDiscoveredUrl()`: Toggles selection of individual URLs
- `selectAllDiscovered()`: Selects all discovered URLs
- `deselectAllDiscovered()`: Deselects all discovered URLs

## User Flow

1. User navigates to the "Add your page links" step in create bot flow
2. User clicks "Discover pages from URL" button
3. Discovery panel appears with:
   - Help text explaining path-based discovery
   - URL input field
   - "Discover" button
4. User enters a URL (e.g., `https://example.com/hotel/tokyo/`)
5. User clicks "Discover" or presses Enter
6. Discovery begins:
   - Loading animation appears
   - URLs stream in real-time
   - Counter shows progress
7. When discovery completes or is stopped:
   - List of discovered URLs appears
   - All URLs are pre-selected
8. User can:
   - Select/deselect individual URLs
   - Use "Select all" / "Deselect all"
   - Click "Add X pages" to add to shared URLs
   - Click "Cancel" to discard
9. Added URLs appear in the main URL rows with auto-generated labels
10. User can edit labels/URLs as needed before training

## Example Use Cases

### Use Case 1: Hotel Chain with Multiple Properties

**Scenario**: A hotel chain has a section for each property on their website.

**Input URL**: `https://hotelchain.com/properties/tokyo-shiodome/`

**Discovered Pages**:
- `https://hotelchain.com/properties/tokyo-shiodome/rooms/`
- `https://hotelchain.com/properties/tokyo-shiodome/dining/`
- `https://hotelchain.com/properties/tokyo-shiodome/amenities/`
- `https://hotelchain.com/properties/tokyo-shiodome/booking/`

**Result**: Only pages for the Tokyo Shiodome property are discovered, not other properties.

### Use Case 2: Salon/Barber Shop on Marketplace

**Scenario**: A salon listed on a marketplace site like HotPepper Beauty.

**Input URL**: `https://beauty.hotpepper.jp/slnH000447147/`

**Discovered Pages**:
- `https://beauty.hotpepper.jp/slnH000447147/stylist/T000826013/`
- `https://beauty.hotpepper.jp/slnH000447147/style/L255896009.html`
- `https://beauty.hotpepper.jp/slnH000447147/coupon/`

**Result**: Only pages for this specific salon are discovered, not other salons on the platform.

### Use Case 3: E-commerce Product Category

**Scenario**: An online store wants to train a bot on a specific product category.

**Input URL**: `https://store.com/products/electronics/laptops/`

**Discovered Pages**:
- `https://store.com/products/electronics/laptops/gaming/`
- `https://store.com/products/electronics/laptops/business/`
- `https://store.com/products/electronics/laptops/ultrabooks/`

**Result**: Only laptop-related pages are discovered, not tablets, phones, or other categories.

## Technical Details

### Discovery Methods

The system supports multiple discovery methods (inherited from the existing implementation):

1. **Auto (Default)**: Combines multiple strategies:
   - URLFinder (if available)
   - Sitemap discovery
   - HTTP crawl with link extraction
   - Crawl4AI browser-based discovery

2. **Sitemap**: Uses the website's sitemap.xml

The "auto" method is used by default as it provides the best coverage.

### Time Limits

- **Client-side timeout**: 60 seconds
- **Server-side timeout**: Configurable via `max_duration_sec` parameter
- Discovery can be stopped by the user at any time

### URL Filtering

Multiple layers of filtering ensure quality results:

1. **Path-based filtering**: `_is_url_under_root_path(url, root_url)`
2. **Robots.txt compliance**: `robots_policy().filter_urls(urls)`
3. **Non-page URL filtering**: Excludes images, CSS, JS, fonts, etc.
4. **Duplicate prevention**: URLs are deduplicated before display

### Performance Optimizations

- **Streaming updates**: URLs appear as they're discovered (not all at once)
- **Auto-selection**: Discovered URLs are pre-selected for user convenience
- **Duplicate prevention**: Existing URLs are not re-added
- **Abort controller**: Discovery can be cancelled without memory leaks

## Testing Recommendations

1. **Test with different URL patterns**:
   - Domain root: `https://example.com/`
   - Single-level path: `https://example.com/blog/`
   - Multi-level path: `https://example.com/products/electronics/laptops/`
   - Path with trailing slash and without

2. **Test edge cases**:
   - Invalid URLs
   - URLs that don't exist (404)
   - Very large sites (many pages)
   - Sites with no discoverable pages
   - Sites blocked by robots.txt

3. **Test user interactions**:
   - Stop discovery mid-stream
   - Select/deselect URLs
   - Add discovered URLs to existing rows
   - Cancel without adding
   - Enter invalid URLs

4. **Test integration**:
   - Verify discovered URLs are properly added to shared URL rows
   - Verify labels are generated correctly
   - Verify training works with discovered URLs
   - Verify no duplicates are created

## Future Enhancements (Optional)

1. **Smart categorization**: Group discovered URLs by type (pricing, services, contact, etc.)
2. **Preview before adding**: Show a preview of page content before adding
3. **Bulk label editing**: Allow users to set labels for multiple URLs at once
4. **Save discovery results**: Cache discovered URLs for later use
5. **Depth control**: Allow users to limit crawl depth
6. **Exclude patterns**: Allow users to exclude certain URL patterns

## Conclusion

This implementation provides a seamless way for users to discover and add pages from a specific section of a website without manually entering each URL. The path-based filtering ensures that only relevant pages within the specified section are discovered, making it perfect for:

- Multi-tenant platforms (like HotPepper Beauty)
- Hotel chains with multiple properties
- E-commerce sites with product categories
- Any website where the bot should focus on a specific section

The implementation leverages existing backend infrastructure and adds a user-friendly frontend interface that integrates seamlessly with the existing create bot flow.
