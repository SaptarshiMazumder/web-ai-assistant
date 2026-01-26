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


async def _run(url: str, *, bucket_name: str, base_prefix: str, corpus_resource: str) -> None:
    from infrastructure.rag.crawl_service import (
        crawl_site_bfs,
        upload_markdown_docs_to_gcs,
        import_gcs_prefix_into_corpus,
        CRAWL_MAX_DEPTH,
        CRAWL_MAX_CONCURRENCY,
    )
    import google.auth
    from google.cloud import storage
    import vertexai

    _emit({"type": "stage", "stage": "starting_browser"})
    # Some crawlers only emit progress after the first successful page; send a heartbeat.
    _emit({"type": "progress", "pages_crawled": 0, "url": url, "depth": 0})
    _emit({"type": "stage", "stage": "crawling"})

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
        docs = await crawl_site_bfs(
            url,
            max_depth=CRAWL_MAX_DEPTH,
            max_concurrent=CRAWL_MAX_CONCURRENCY,
            stop_event=None,
            progress_cb=_on_progress,
        )
    except Exception as e:
        _emit({"type": "error", "error": str(e)})
        raise

    _emit({"type": "result", "docs_count": int(len(docs or []))})

    if not docs:
        _emit({"type": "stage", "stage": "done"})
        return

    _emit({"type": "stage", "stage": "uploading"})

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set for worker")

    # Load credentials explicitly (avoid picking up ADC from a different user).
    creds, proj = google.auth.load_credentials_from_file(creds_path)
    creds_type = "service_account" if getattr(creds, "service_account_email", None) else "non_service_account"
    _emit({"type": "auth", "creds_path": creds_path, "creds_type": creds_type, "project": proj})

    storage_client = storage.Client(credentials=creds, project=proj)
    gcs_prefix = upload_markdown_docs_to_gcs(
        bucket_name=bucket_name,
        base_prefix=base_prefix,
        docs=docs,
        storage_client=storage_client,
    )
    _emit({"type": "gcs_prefix", "gcs_prefix": gcs_prefix})

    _emit({"type": "stage", "stage": "importing"})
    # Ensure Vertex SDK uses the same credentials.
    try:
        vertexai.init(project=proj, location=os.environ.get("LOCATION", "us-central1"), credentials=creds)
    except Exception:
        # If init fails, import may still work via env; surface in logs if it errors later.
        pass
    import_gcs_prefix_into_corpus(corpus_resource=corpus_resource, bucket_name=bucket_name, prefix=gcs_prefix)

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
    parser.add_argument("--url", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--creds", required=False, default="")
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
        asyncio.run(_run(args.url, bucket_name=args.bucket, base_prefix=args.base_prefix, corpus_resource=args.corpus))
        return 0
    except Exception as e:
        _emit({"type": "error", "error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
