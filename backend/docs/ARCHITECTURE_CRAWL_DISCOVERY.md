# Clean Architecture: Crawling & URL Discovery

## Summary

- **Crawling** is behind a domain port; swapping the crawler (e.g. replace crawl4ai) is done by implementing the port and wiring it in DI.
- **URL discovery** is now behind a domain port; swapping discovery (e.g. replace crawl4ai discovery) is done by implementing the port and wiring it in DI.

## Crawling (already clean)

| Layer | What | Swap how |
|-------|------|----------|
| **Domain** | `CrawlerRepository` (Protocol) in `domain/repositories.py` — `crawl_urls_bfs`, `crawl_urls_list`, `discover_internal_urls` | N/A (port) |
| **Application** | `IndexingService` depends on `CrawlerRepository`; gets implementation via DI | N/A |
| **Infrastructure** | `Crawl4AICrawlerRepository` in `infrastructure/repositories/crawl4ai_crawler_repository.py` | Replace with another adapter (e.g. `ScrapyCrawlerRepository`) |
| **DI** | `common/di/container.py`: `crawler_repo = Crawl4AICrawlerRepository()` passed to `IndexingService` | Change to `crawler_repo = YourCrawlerRepository()` |

So: to use a different crawling service, implement `CrawlerRepository` and register it in `container.py` (and in the worker/tasks if they construct the repo directly).

## URL discovery (now clean)

| Layer | What | Swap how |
|-------|------|----------|
| **Domain** | `UrlDiscoveryPort` (Protocol) in `domain/repositories.py` — `discover(root_url, method)`, `discover_stream(root_url, method, ...)` | N/A (port) |
| **API** | Routes call `url_discovery().discover()` / `url_discovery().discover_stream()`; no import of crawl_service or url_discovery_service | N/A |
| **Infrastructure** | `Crawl4AIUrlDiscoveryAdapter` in `infrastructure/rag/url_discovery_adapter.py` — delegates to crawl_service and url_discovery_service | Replace with another adapter (e.g. `CustomUrlDiscoveryAdapter`) |
| **DI** | `common/di/container.py`: `url_discovery()` returns `Crawl4AIUrlDiscoveryAdapter()` | Change to `return YourUrlDiscoveryAdapter()` |

So: to use a different discovery service, implement `UrlDiscoveryPort` and in `container.py` return that implementation from `url_discovery()`.

## Remaining coupling (optional to fix later)

- **Worker** (`infrastructure/workers/worker_index_job.py`) still constructs `Crawl4AICrawlerRepository()` and imports `CRAWL_MAX_DEPTH`, `CRAWL_MAX_CONCURRENCY` from `crawl_service`. To make the worker fully swappable, it could receive the crawler repo (and config) via CLI/env or a small DI.
- **Crawl4AICrawlerRepository** and **crawl_service** still share crawl4ai-specific config (e.g. `HEADLESS`, `CRAWL_MAX_DEPTH`). Moving those into a dedicated config module or passing them into the repository would reduce coupling further.
- **CrawlerRepository.discover_internal_urls** is in the protocol but not used by the API (the API uses `UrlDiscoveryPort.discover_stream`). It is still used by `url_discovery_service.discover_urls_auto` indirectly (via `crawl_service.discover_internal_urls`). No change required for swapping.
