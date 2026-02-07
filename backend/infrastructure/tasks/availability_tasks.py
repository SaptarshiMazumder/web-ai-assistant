import asyncio
import base64
import json
import os
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from celery import Task
from openai import OpenAI
from playwright.async_api import async_playwright

from infrastructure.celery_app import celery_app
from infrastructure.db.repositories import PostgresAvailabilityJobRepository


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_url_from_png(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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


def _extract_output_list(response: Any) -> List[Dict[str, Any]]:
    if response is None:
        return []
    if isinstance(response, dict):
        return response.get("output") or []
    output = getattr(response, "output", None)
    if output is None:
        try:
            dumped = response.model_dump()  # type: ignore[attr-defined]
            return dumped.get("output") or []
        except Exception:
            return []
    return output


def _extract_text_output(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        return response.get("output_text") or response.get("text") or ""
    text = getattr(response, "output_text", None)
    if text:
        return text
    text = getattr(response, "text", None)
    if text:
        return text
    try:
        dumped = response.model_dump()  # type: ignore[attr-defined]
        return dumped.get("output_text") or dumped.get("text") or ""
    except Exception:
        return ""


def _find_computer_call(output_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in output_items:
        if isinstance(item, dict) and item.get("type") == "computer_call":
            return item
    return None


def _extract_pending_safety_checks(call_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = call_item.get("pending_safety_checks")
    if isinstance(checks, list):
        return checks
    return []


def _prompt_for_availability(
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
) -> str:
    return (
        "You are a hotel availability assistant. Use the browser to check availability on the hotel page.\n"
        f"URL: {url}\n"
        f"Check-in: {check_in}\n"
        f"Check-out: {check_out}\n"
        f"Adults: {adults}\n"
        f"Children: {children}\n"
        f"Rooms: {rooms}\n\n"
        "Instructions:\n"
        "- Interact with the page to set dates and guests.\n"
        "- Click the availability/search button.\n"
        "- Wait for results.\n"
        "- Summarize availability and pricing in plain English.\n"
        "- If blocked or unavailable, explain briefly.\n"
    )


def _truncate_text(text: str, max_chars: int = 6000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _build_llm_prompt(
    *,
    question: str,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    extracted_text: str,
) -> str:
    user_question = question.strip() or "Summarize the availability and pricing for the selected dates."
    return (
        "You are a hotel availability assistant. Answer the user's question using ONLY the extracted text.\n"
        "If availability or pricing is missing, say you could not find it.\n"
        "Keep the answer concise and structured.\n\n"
        f"User question: {user_question}\n"
        f"URL: {url}\n"
        f"Check-in: {check_in}\n"
        f"Check-out: {check_out}\n"
        f"Adults: {adults}\n"
        f"Children: {children}\n"
        f"Rooms: {rooms}\n\n"
        "Extracted text:\n"
        f"{_truncate_text(extracted_text)}\n"
    )


def _answer_with_llm(
    *,
    question: str,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    extracted_text: str,
) -> Optional[str]:
    model = (os.environ.get("AVAILABILITY_SUMMARY_MODEL") or "gpt-4o-mini").strip()
    if not model or model.lower() in ("none", "disabled", "off"):
        return None
    if not extracted_text:
        return None
    try:
        client = OpenAI()
        prompt = _build_llm_prompt(
            question=question,
            url=url,
            check_in=check_in,
            check_out=check_out,
            adults=adults,
            children=children,
            rooms=rooms,
            extracted_text=extracted_text,
        )
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            truncation="auto",
        )
        answer = _extract_text_output(response).strip()
        return answer or None
    except Exception:
        return None


def _with_booking_params(
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
) -> str:
    parsed = urlparse(url)
    original_params = parse_qs(parsed.query)
    params: Dict[str, List[str]] = {}
    # Keep only a minimal set of optional tracking/language params.
    for key in ("lang", "label", "aid", "sid", "selected_currency", "ac_lang"):
        if key in original_params and original_params[key]:
            params[key] = [original_params[key][0]]
    params["checkin"] = [check_in]
    params["checkout"] = [check_out]
    params["group_adults"] = [str(max(1, adults))]
    params["group_children"] = [str(max(0, children))]
    params["no_rooms"] = [str(max(1, rooms))]
    if children > 0:
        params["age"] = [str(10)] * children
    query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=query, fragment=""))


async def _extract_booking_summary(page) -> str:
    try:
        body_text = await page.inner_text("body")
        lowered = body_text.lower()
        if "no availability" in lowered or "no rooms available" in lowered:
            return "No availability found for the selected dates."
    except Exception:
        body_text = ""

    room_names: List[str] = []
    prices: List[str] = []
    try:
        room_names = [
            t.strip()
            for t in await page
            .locator('#hprt-form [data-testid="room-name"], #hprt-form .hprt-roomtype-name, [data-testid="room-name"]')
            .all_inner_texts()
            if t.strip()
        ]
    except Exception:
        room_names = []
    try:
        prices = [
            t.strip()
            for t in await page
            .locator('#hprt-form [data-testid="price-and-discounted-price"], #hprt-form [data-testid="price"], #hprt-form .hprt-price-price, [data-testid="price-and-discounted-price"], [data-testid="price"]')
            .all_inner_texts()
            if t.strip()
        ]
    except Exception:
        prices = []

    lines = []
    if room_names or prices:
        count = max(len(room_names), len(prices))
        for i in range(min(count, 5)):
            name = room_names[i] if i < len(room_names) else "Room option"
            price = prices[i] if i < len(prices) else "Price not shown"
            lines.append(f"- {name}: {price}")
        return "Availability summary:\n" + "\n".join(lines)

    # Look for explicit sold-out or availability messages in the availability section.
    try:
        availability_text = await page.locator('#availability, #hprt-form, #room_availability_container').inner_text()
        lower_avail = availability_text.lower()
        if "sold out" in lower_avail or "no availability" in lower_avail or "no rooms available" in lower_avail:
            return "No availability found for the selected dates."
    except Exception:
        pass

    try:
        availability_text = await page.locator(
            "#availability, #hprt-form, #room_availability_container"
        ).inner_text()
        if availability_text.strip():
            snippet = availability_text[:1200].strip()
            return f"Availability details (raw text excerpt):\n{snippet}"
    except Exception:
        pass

    if body_text:
        snippet = body_text[:1200].strip()
        return f"Availability details (raw text excerpt):\n{snippet}"
    return "Could not extract availability details from the page."


async def _run_booking_adapter(
    *,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    max_seconds: int,
    screenshots_dir: str,
) -> Tuple[str, int, str, str]:
    width = int(os.environ.get("AVAILABILITY_VIEWPORT_WIDTH", "1365"))
    height = int(os.environ.get("AVAILABILITY_VIEWPORT_HEIGHT", "768"))
    headless = (os.environ.get("AVAILABILITY_HEADLESS") or "false").strip().lower() in ("true", "1", "yes")
    user_agent = os.environ.get("AVAILABILITY_USER_AGENT") or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    start = time.time()
    step_index = 0

    async def safe_goto(page, target_url: str) -> None:
        try:
            await page.goto(target_url, wait_until="commit", timeout=60000)
            await page.wait_for_load_state("domcontentloaded", timeout=60000)
        except Exception as e:
            if "net::ERR_ABORTED" in str(e):
                # Booking.com sometimes triggers an immediate redirect that aborts the original navigation.
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
                return
            raise

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            user_agent=user_agent,
        )
        page = await context.new_page()

        # Use URL as-is when user pastes full booking URL (with dates/guests already in query); otherwise build params
        if check_in and check_out:
            booking_url = _with_booking_params(url, check_in, check_out, adults, children, rooms)
        else:
            booking_url = url
        await safe_goto(page, booking_url)
        if time.time() - start > max_seconds:
            await browser.close()
            return "Timed out while loading the page.", step_index
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        await page.wait_for_timeout(1200)
        png = await page.screenshot(full_page=False)
        _write_step_screenshot(screenshots_dir, step_index, png)
        step_index += 1

        # Accept cookies if present
        for selector in ["#onetrust-accept-btn-handler", 'button:has-text("Accept")', 'button:has-text("I Agree")']:
            try:
                if await page.locator(selector).first.is_visible(timeout=2000):
                    await page.locator(selector).first.click()
                    break
            except Exception:
                continue

        # The booking URL is already loaded; avoid re-navigation unless redirected off-host.
        if urlparse(page.url).hostname != urlparse(booking_url).hostname:
            await safe_goto(page, booking_url)
            if time.time() - start > max_seconds:
                await browser.close()
                return "Timed out while loading availability.", step_index
            png = await page.screenshot(full_page=False)
            _write_step_screenshot(screenshots_dir, step_index, png)
            step_index += 1

        # Try to jump to the availability section / open availability tab.
        for selector in [
            'a[href*="#availability"]',
            'a:has-text("Availability")',
            'a:has-text("Rooms")',
            'button:has-text("See availability")',
            'button:has-text("Check availability")',
        ]:
            try:
                if await page.locator(selector).first.is_visible(timeout=1500):
                    await page.locator(selector).first.click()
                    break
            except Exception:
                continue

        try:
            await page.evaluate(
                "() => document.querySelector('#availability, #hprt-form')?.scrollIntoView({behavior: 'instant'})"
            )
            await page.wait_for_timeout(1200)
            await page.mouse.wheel(0, 1200)
            await page.wait_for_timeout(800)
        except Exception:
            pass

        # Wait for some availability indicators
        selectors = [
            '[data-testid="room-name"]',
            '[data-testid="price-and-discounted-price"]',
            '#hprt-form',
            '#room_availability_container',
        ]
        for sel in selectors:
            try:
                await page.wait_for_selector(sel, timeout=30000)
                break
            except Exception:
                continue
        try:
            await page.wait_for_function(
                "() => document.querySelectorAll('[data-testid=\"room-name\"],"
                " '[data-testid=\"price-and-discounted-price\"],"
                " '#hprt-form .hprt-roomtype-name').length > 0",
                timeout=20000,
            )
        except Exception:
            pass

        png = await page.screenshot(full_page=False)
        _write_step_screenshot(screenshots_dir, step_index, png)
        step_index += 1

        summary = await _extract_booking_summary(page)
        raw_text = ""
        raw_html = ""
        try:
            raw_text = await page.inner_text("body")
        except Exception:
            raw_text = ""
        try:
            raw_html = await page.content()
        except Exception:
            raw_html = ""
        await browser.close()

    return summary, step_index, raw_text, raw_html


async def _run_agent(
    *,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    max_seconds: int,
    screenshots_dir: str,
) -> Tuple[str, int, str, str]:
    client = OpenAI()
    width = int(os.environ.get("AVAILABILITY_VIEWPORT_WIDTH", "1365"))
    height = int(os.environ.get("AVAILABILITY_VIEWPORT_HEIGHT", "768"))
    headless = (os.environ.get("AVAILABILITY_HEADLESS") or "false").strip().lower() in ("true", "1", "yes")
    max_steps = int(os.environ.get("AVAILABILITY_MAX_STEPS", "25"))

    start = time.time()
    step_index = 0
    summary = ""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        png = await page.screenshot(full_page=False)
        _write_step_screenshot(screenshots_dir, step_index, png)
        step_index += 1

        prompt = _prompt_for_availability(url, check_in, check_out, adults, children, rooms)
        response = client.responses.create(
            model="computer-use-preview",
            tools=[
                {
                    "type": "computer_use_preview",
                    "display_width": width,
                    "display_height": height,
                    "environment": "browser",
                }
            ],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _data_url_from_png(png)},
                    ],
                }
            ],
            truncation="auto",
        )

        prev_response_id = getattr(response, "id", None) or (response.get("id") if isinstance(response, dict) else None)

        for _ in range(max_steps):
            if time.time() - start > max_seconds:
                summary = summary or "Timed out while checking availability."
                break

            output_items = _extract_output_list(response)
            call = _find_computer_call(output_items)
            if not call:
                summary = _extract_text_output(response)
                break

            call_id = call.get("id")
            if not call_id:
                summary = _extract_text_output(response) or "Agent returned no actionable step."
                break

            action = call.get("action") or {}
            action_type = action.get("type")

            if action_type == "click":
                await page.mouse.click(action.get("x", 0), action.get("y", 0))
            elif action_type == "double_click":
                await page.mouse.dblclick(action.get("x", 0), action.get("y", 0))
            elif action_type == "scroll":
                await page.mouse.wheel(0, action.get("scroll_y", 600))
            elif action_type == "type":
                await page.keyboard.type(action.get("text", ""))
            elif action_type == "keypress":
                keys = action.get("keys") or []
                for key in keys:
                    await page.keyboard.press(key)
            elif action_type == "wait":
                await page.wait_for_timeout(int(action.get("duration_ms", 1000)))
            elif action_type == "move":
                await page.mouse.move(action.get("x", 0), action.get("y", 0))
            else:
                # Unknown action: wait a bit to avoid tight loop
                await page.wait_for_timeout(500)

            png = await page.screenshot(full_page=False)
            _write_step_screenshot(screenshots_dir, step_index, png)
            step_index += 1

            pending_checks = _extract_pending_safety_checks(call)
            response = client.responses.create(
                model="computer-use-preview",
                tools=[
                    {
                        "type": "computer_use_preview",
                        "display_width": width,
                        "display_height": height,
                        "environment": "browser",
                    }
                ],
                input=[
                    {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": _data_url_from_png(png),
                        },
                        "current_url": page.url,
                        "acknowledged_safety_checks": pending_checks,
                    }
                ],
                truncation="auto",
                previous_response_id=prev_response_id,
            )
            prev_response_id = getattr(response, "id", None) or (response.get("id") if isinstance(response, dict) else None)

        raw_text = ""
        raw_html = ""
        try:
            raw_text = await page.inner_text("body")
        except Exception:
            raw_text = ""
        try:
            raw_html = await page.content()
        except Exception:
            raw_html = ""
        await browser.close()

    return summary or "No summary returned.", step_index, raw_text, raw_html


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

    try:
        host = urlparse(url).hostname or ""
        if host.endswith("booking.com"):
            summary, steps, raw_text, raw_html = asyncio.run(
                _run_booking_adapter(
                    url=url,
                    check_in=check_in,
                    check_out=check_out,
                    adults=adults,
                    children=children,
                    rooms=rooms,
                    max_seconds=max_seconds,
                    screenshots_dir=screenshots_dir,
                )
            )
        else:
            summary, steps, raw_text, raw_html = asyncio.run(
                _run_agent(
                    url=url,
                    check_in=check_in,
                    check_out=check_out,
                    adults=adults,
                    children=children,
                    rooms=rooms,
                    max_seconds=max_seconds,
                    screenshots_dir=screenshots_dir,
                )
            )
        raw_max = int(os.environ.get("AVAILABILITY_RAW_MAX_CHARS", "0") or 0)
        if raw_max > 0:
            raw_text = raw_text[:raw_max]
            raw_html = raw_html[:raw_max]
        raw_text_path = _write_text_file(screenshots_dir, "extracted.txt", raw_text)
        raw_html_path = _write_text_file(screenshots_dir, "page.html", raw_html)
        llm_answer = _answer_with_llm(
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
        job.status = "done"
        job.summary = summary
        job.steps_count = steps
        job.raw_text_path = raw_text_path
        job.raw_html_path = raw_html_path
        repo.update(job)
        return {"status": "done", "summary": summary}
    except Exception as exc:
        job.status = "error"
        job.last_error = str(exc)[:300]
        repo.update(job)
        return {"status": "error", "error": job.last_error}
