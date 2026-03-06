# Platform config

## platform_profiles.yml

Single source of truth for all business-type-specific behavior (Tabelog, HotPepper, TableCheck, etc.).

**Edit this file to add or change platforms** — no code changes needed.

### Schema

| Section | Description |
|---------|-------------|
| `reservation_platform_config` | Maps platform_id (tabelog, hotpepper, tablecheck) → `widget_key`, `domain_key`, `url_placeholder`, `knowledge_tabs` |
| `default_asset_rules` | `marker_rule`, `evidence_template` — fallback when no platform defines `asset_rules` |
| `default_asset_term_config` | `generic_tokens`, `asset_intent_terms`, `visual_request_terms`, `visual_request_many_terms`, `visual_suppress_terms` — used for asset matching/scoring |
| `default_suggested_messages` | Fallback when no platform defines `suggested_messages` |
| `platforms` | Per-domain config keyed by domain (e.g. `tabelog.com`) |

### Platform entry fields

| Field | Description |
|-------|-------------|
| `domain_pattern` | Regex to match URLs (e.g. `tabelog\.com`) |
| `include_paths` | URL path patterns to include in crawl |
| `exclude_paths` | URL path patterns to exclude |
| `priority` | Tiebreaker when multiple profiles match |
| `strip_query_params` | Remove `?` params for URL deduplication |
| `menu_url_patterns` | Paths containing menu/course data |
| `menu_extraction_rules` | `enabled`, `extractor`, `path_category_patterns`, etc. |
| `metadata` | Platform-specific prompts and rules |

### reservation_platform_config entry fields

| Field | Description |
|-------|-------------|
| `widget_key` | Key in widget_config for the platform URL (e.g. `tabelogUrl`) |
| `domain_key` | Domain key for platform profiles (e.g. `tabelog.com`) |
| `url_placeholder` | Placeholder URL for the dashboard input |
| `knowledge_tabs` | Dashboard tabs to show: `[menu]` for Menu tab only (Tabelog, HotPepper), `[image]` for Image tab only (default) |

### metadata sub-fields

| Field | Description |
|-------|-------------|
| `reservation` | `enabled`, `link_label` (en/ja), `instruction_template` (en/ja) |
| `suggested_messages` | Quick-reply options (Menu, Reservation, etc.) |
| `asset_instructions` | LLM prompt rules for when to show menu assets vs reservation link |
| `asset_rules` | `marker_rule`, `evidence_template`, `asset_term_config` — asset bank instruction, evidence format, and term config (overrides default) |
