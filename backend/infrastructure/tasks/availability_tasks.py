import asyncio
import json
import os
from typing import Any, Dict, Optional

from celery import Task

from infrastructure.availability.extraction import run_generic_extraction
from infrastructure.availability.logging_utils import (
    get_logger,
    setup_availability_file_logging,
    teardown_availability_file_logging,
)
from infrastructure.availability.llm_extraction import extract_availability_summary
from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresAvailabilityJobRepository


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _write_step_screenshot(dir_path: str, step_index: int, png_bytes: bytes) -> str:
    _ensure_dir(dir_path)
    filename = f"step_{step_index:03d}.png"
    path = os.path.join(dir_path, filename)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path


def _write_text_file(dir_path: str, filename: str, content: str) -> Optional[str]:
    if not content:
        return None
    _ensure_dir(dir_path)
    path = os.path.join(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@celery_app.task(
    name="infrastructure.tasks.availability_tasks.availability_job",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def availability_job_task(
    self: Task,
    job_id: str,
    bot_id: str,
    org_id: str,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    max_seconds: int,
    screenshots_dir: str,
    question: str = "",
) -> Dict[str, Any]:
    repo = PostgresAvailabilityJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        return {"status": "error", "error": "job not found"}

    job.status = "running"
    job.celery_task_id = self.request.id
    repo.update(job)

    log = get_logger()
    setup_availability_file_logging(screenshots_dir)
    log.info("=== availability_job started ===")
    log.info("url=%s check_in=%s check_out=%s adults=%d children=%d rooms=%d",
             url[:120] if url else "", check_in, check_out, adults, children, rooms)
    debug_path = os.path.join(screenshots_dir, "debug_info.json")
    if os.path.exists(debug_path):
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_info = json.load(f)
            log.info("debug_info from API: %s", json.dumps(debug_info, default=str)[:500])
        except Exception as e:
            log.warning("could not read debug_info: %s", e)

    def write_screenshot(step_index: int, png_bytes: bytes) -> str:
        return _write_step_screenshot(screenshots_dir, step_index, png_bytes)

    try:
        summary, steps, raw_text, raw_html = asyncio.run(
            run_generic_extraction(
                url=url,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                rooms=rooms,
                max_seconds=max_seconds,
                write_screenshot=write_screenshot,
            )
        )

        raw_max = int(os.environ.get("AVAILABILITY_RAW_MAX_CHARS", "0") or 0)
        if raw_max > 0:
            raw_text = raw_text[:raw_max]
            raw_html = raw_html[:raw_max]
        raw_text_path = _write_text_file(screenshots_dir, "extracted.txt", raw_text)
        raw_html_path = _write_text_file(screenshots_dir, "page.html", raw_html)

        log.info("extraction done: raw_text len=%d, passing to LLM", len(raw_text or ""))
        llm_answer = extract_availability_summary(
            question=question,
            url=url,
            check_in=check_in,
            check_out=check_out,
            adults=adults,
            children=children,
            rooms=rooms,
            extracted_text=raw_text or summary,
        )
        if llm_answer:
            summary = llm_answer
        else:
            log.warning("LLM returned no answer, using raw summary")

        log.info("=== availability_job done, summary len=%d ===", len(summary or ""))
        teardown_availability_file_logging(screenshots_dir)

        job.status = "done"
        job.summary = summary
        job.steps_count = steps
        job.raw_text_path = raw_text_path
        job.raw_html_path = raw_html_path
        repo.update(job)
        return {"status": "done", "summary": summary}
    except Exception as exc:
        log = get_logger()
        log.exception("availability_job failed: %s", exc)
        teardown_availability_file_logging(screenshots_dir)
        job.status = "error"
        job.last_error = str(exc)[:300]
        repo.update(job)
        return {"status": "error", "error": job.last_error}
