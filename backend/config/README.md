# Platform config

## platform_profiles.yml

Single source of truth for business/platform behavior (Tabelog, HotPepper, TableCheck, etc.).

Edit this file to add or change platform behavior. No code change should be required for runtime settings.

### Top-level sections

| Section | Description |
|---------|-------------|
| `default_post_crawl_jobs` | Default jobs after crawl (`topic_extraction`, `booking_link`) |
| `defaults` | Global defaults for source language, menu aliases/tokens, prompts, function mapping, and stopwords |
| `reservation_platform_config` | Platform map (`platform_id -> widget_key/domain_key/url_placeholder/knowledge_tabs/post_crawl_jobs`) |
| `default_asset_rules` | Default `marker_rule` and `evidence_template` |
| `default_asset_term_config` | Default term lists for asset matching |
| `default_suggested_messages` | Fallback suggested messages |
| `platforms` | Per-domain config keyed by domain (e.g., `tabelog.com`) |

### reservation_platform_config fields

| Field | Description |
|-------|-------------|
| `widget_key` | Key in widget config for URL (e.g., `tabelogUrl`) |
| `domain_key` | Domain key in `platforms` |
| `url_placeholder` | Placeholder URL shown in dashboard |
| `knowledge_tabs` | Dashboard tabs to show (e.g., `[menu]`, `[image]`) |
| `post_crawl_jobs` | Platform-specific post-crawl jobs |

### Widget config reservation URL shape

- Canonical internal shape: `reservation_links: { platform_id: url }`
- Backward compatible (still supported): `tabelogUrl`, `hotPepperUrl`, `tableCheckUrl`

### platform metadata fields

| Field | Description |
|-------|-------------|
| `reservation` | Reservation enablement, labels, and instruction template |
| `suggested_messages` | Suggested action chips/buttons |
| `asset_instructions` | Prompt guidance for asset use |
| `asset_rules` | Marker/evidence and term-config overrides |
