# Platform config

## platform_profiles.yml

Single source of truth for business/platform behavior (Tabelog, HotPepper, TableCheck, etc.).

Edit this file to add or change platform behavior. No code change should be required for runtime settings.

### Top-level sections

| Section | Description |
|---------|-------------|
| `default_post_crawl_jobs` | Legacy compatibility list (pipeline is canonical execution path) |
| `job_pipeline` | Canonical sequential job pipeline (catalog, workflows, gates) |
| `reservation_url_rules` | Platform URL auto-fill rules for reservation URL job |
| `defaults` | Global defaults for source language, menu aliases/tokens, prompts, function mapping, and stopwords |
| `reservation_platform_config` | Platform map (`platform_id -> widget_key/domain_key/url_placeholder/knowledge_tabs/post_crawl_jobs`) |
| `default_asset_rules` | Default `marker_rule` and `evidence_template` |
| `default_asset_term_config` | Default term lists for asset matching |
| `default_suggested_messages` | Fallback suggested messages |
| `platforms` | Per-domain config keyed by domain (e.g., `tabelog.com`) |

### dashboard.create_bot_flow shape

- `step_groups`: ordered sidebar groups (`id`, localized `label`, localized `description`)
- `screen_definitions`: screen map keyed by screen id
  - required: `path`, `step_group`, `component`
  - optional (for `action_destination_url`): `action_key`, localized `title/subtitle/field_*`
  - optional `visibility`:
    - `business_types`: only show for matching business types
    - `requires_selected_reservation_platform`: only show after a reservation platform is selected
    - `reservation_platform_ids`: only show when selected reservation platform id is in this list
    - `requires_workflow_steps`: only show when resolved workflow contains these job ids
- `screen_order`: ordered list of screen ids to render in the flow

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

### job_pipeline shape

- `job_pipeline.jobs`: job catalog (`job_id -> runner_ref/retries/timeout/on_failure/dashboard_label`)
- `job_pipeline.workflows.default`: ordered job list for generic bots
- `job_pipeline.workflows.platform_overrides`: ordered job list per platform id (e.g. `tabelog`, `hotpepper`)
- `job_pipeline.gates`: optional pause/resume gates (`step_id`, `pause_on_step`, `resume_event`)

The pipeline engine executes steps sequentially in YAML order and applies each step's failure policy (`continue` or `stop`).

### reservation_url_rules shape

- Per platform id (e.g. `tabelog`, `hotpepper`)
- `assignment_mode`: `base_url` or `candidate`
- `base_url_source_key`: context key used by `base_url` mode (default `root_url`)
- `base_path_pattern`: optional regex to normalize shop base path in `base_url` mode
- `allowed_domains`: allowed hostnames
- `include_path_patterns`: required regex path patterns
- `exclude_path_patterns`: deny regex path patterns
- `path_keyword_scores`: score map used to pick best candidate URL

### platform metadata fields

| Field | Description |
|-------|-------------|
| `reservation` | Reservation enablement, labels, and instruction template |
| `suggested_messages` | Suggested action chips/buttons |
| `asset_instructions` | Prompt guidance for asset use |
| `asset_rules` | Marker/evidence and term-config overrides |
