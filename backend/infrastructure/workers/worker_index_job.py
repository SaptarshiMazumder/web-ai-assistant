import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional


_EVENT_PREFIX = "WEB_AI_EVENT "


def _emit(obj: Dict[str, Any]) -> None:
    # Prefix events so the API can parse them even if crawler logs are noisy.
    sys.stdout.write(_EVENT_PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def _run(
    url: str | None,
    urls: list[str] | None,
    *,
    bucket_name: str,
    base_prefix: str,
    corpus_resource: str,
    headless: Optional[bool],
) -> None:
    from infrastructure.rag.crawl_service import CRAWL_MAX_DEPTH, CRAWL_MAX_CONCURRENCY
    from infrastructure.repositories import Crawl4AICrawlerRepository, GCSDocumentStorageRepository, VertexRAGRepository
    from google.cloud import storage
    import google.auth
    import vertexai

    _emit({"type": "stage", "stage": "starting_browser"})
    # Some crawlers only emit progress after the first successful page; send a heartbeat.
    _emit({"type": "progress", "pages_crawled": 0, "url": (url or ""), "depth": 0})
    _emit({"type": "stage", "stage": "crawling"})

    crawler_repo = Crawl4AICrawlerRepository()

    def _on_progress(evt: Dict[str, Any]):
        if evt.get("type") == "page_crawled":
            _emit(
                {
                    "type": "progress",
                    "pages_crawled": int(evt.get("count") or 0),
                    "url": str(evt.get("url") or ""),
                    "depth": int(evt.get("depth") if evt.get("depth") is not None else -1),
                }
            )
        elif evt.get("type") == "fetch":
            # Surface per-fetch diagnostics so API runs can be debugged via /status.
            _emit(
                {
                    "type": "fetch",
                    "url": str(evt.get("url") or ""),
                    "success": bool(evt.get("success")),
                    "status_code": evt.get("status_code"),
                    "error": str(evt.get("error") or ""),
                    "content_source": str(evt.get("content_source") or ""),
                    "markdown_len": int(evt.get("markdown_len") or 0),
                    "text_len": int(evt.get("text_len") or 0),
                    "extracted_text_len": int(evt.get("extracted_text_len") or 0),
                    "cleaned_html_len": int(evt.get("cleaned_html_len") or 0),
                    "html_len": int(evt.get("html_len") or 0),
                    "raw_html_len": int(evt.get("raw_html_len") or 0),
                }
            )

    try:
        if urls:
            docs = await crawler_repo.crawl_urls_list(
                urls,
                max_concurrent=CRAWL_MAX_CONCURRENCY,
                progress_cb=_on_progress,
                headless=headless,
            )
        else:
            # STRICT MODE: only crawl the explicit URL provided (no BFS expansion).
            docs = await crawler_repo.crawl_urls_list(
                [url or ""],
                max_concurrent=CRAWL_MAX_CONCURRENCY,
                progress_cb=_on_progress,
                headless=headless,
            )
    except Exception as e:
        _emit({"type": "error", "error": str(e)})
        raise

    _emit({"type": "result", "docs_count": int(len(docs or []))})

    if not docs:
        _emit({"type": "stage", "stage": "done"})
        return

    _emit({"type": "stage", "stage": "uploading"})

    from common.gcp_auth import load_gcp_credentials
    creds, proj = load_gcp_credentials()
    creds_type = "service_account" if getattr(creds, "service_account_email", None) else "adc"
    _emit({"type": "auth", "creds_type": creds_type, "project": proj})

    # Extract bot_id from base_prefix (format: <tenant>/bots/<bot_id> or just <tenant>/bots/<bot_id>)
    # base_prefix already includes tenant/bots/bot_id, so we just need the bot_id part
    bot_id = ""
    if "/bots/" in base_prefix:
        parts = base_prefix.split("/bots/")
        if len(parts) > 1:
            bot_id = parts[1].split("/")[0]
    
    # For storage, we need to pass the base_prefix that includes tenant/bots/bot_id
    # The storage repo will append the host prefix and timestamp
    # Use the same credentials as loaded for Vertex AI
    storage_client = storage.Client(credentials=creds, project=proj)
    storage_repo = GCSDocumentStorageRepository(bucket_name, base_prefix, storage_client=storage_client)
    gcs_prefix = storage_repo.save_documents(bot_id, docs)
    _emit({"type": "gcs_prefix", "gcs_prefix": gcs_prefix})

    _emit({"type": "stage", "stage": "importing"})
    # Ensure Vertex SDK uses the same credentials.
    try:
        vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
    except Exception:
        # If init fails, import may still work via env; surface in logs if it errors later.
        pass
    rag_repo = VertexRAGRepository()
    rag_repo.import_documents(corpus_resource, gcs_prefix)

    # Vertex import is async on the Google side.
    _emit({"type": "stage", "stage": "import_submitted"})


def main() -> int:
    # Ensure Playwright subprocess spawning works on Windows.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        # Prevent UnicodeEncodeError when crawl4ai/Playwright logs include unicode
        # (e.g. arrows) and stdout is using a legacy codepage.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=False, default="")
    parser.add_argument("--urls-json", required=False, default="")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--creds", required=False, default="")
    parser.add_argument("--headless", required=False, default="")
    args = parser.parse_args()

    try:
        if args.creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.creds

        # Emit runtime info so the API can prove which interpreter ran the worker.
        _emit(
            {
                "type": "runtime",
                "sys_executable": sys.executable,
                "python_version": sys.version,
                "cwd": os.getcwd(),
                "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            }
        )
        url_list = []
        if args.urls_json:
            try:
                parsed = json.loads(args.urls_json)
                if isinstance(parsed, list):
                    url_list = [str(item) for item in parsed if str(item).strip()]
            except Exception:
                url_list = []
        if not args.url and not url_list:
            raise RuntimeError("Missing url(s) for crawl")

        headless_arg = (args.headless or "").strip().lower()
        headless_value: Optional[bool]
        if headless_arg in ("true", "1", "yes"):
            headless_value = True
        elif headless_arg in ("false", "0", "no"):
            headless_value = False
        else:
            headless_value = None

        asyncio.run(
            _run(
                args.url or None,
                url_list or None,
                bucket_name=args.bucket,
                base_prefix=args.base_prefix,
                corpus_resource=args.corpus,
                headless=headless_value,
            )
        )
        return 0
    except Exception as e:
        _emit({"type": "error", "error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
