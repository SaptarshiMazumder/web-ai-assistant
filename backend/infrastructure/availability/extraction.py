"""Generic availability extraction via Playwright + LLM. Works for any booking site."""

import os
import time
from typing import Callable, Optional, Tuple

from playwright.async_api import async_playwright

from infrastructure.availability.logging_utils import get_logger


async def run_generic_extraction(
    *,
    url: str,
    check_in: str,
    check_out: str,
    adults: int,
    children: int,
    rooms: int,
    max_seconds: int,
    write_screenshot: Optional[Callable[[int, bytes], str]] = None,
) -> Tuple[str, int, str, str]:
    """
    Load booking URL, get page text, return (summary_placeholder, step_index, raw_text, raw_html).
    The caller should pass raw_text to an LLM for extraction - this function does not call LLM.
    Returns a brief raw summary if no LLM is used; typically the caller runs _answer_with_llm
    on raw_text to get the real summary.
    """
    width = int(os.environ.get("AVAILABILITY_VIEWPORT_WIDTH", "1365"))
    height = int(os.environ.get("AVAILABILITY_VIEWPORT_HEIGHT", "768"))
    # Default True so it works in API container (no display). Celery worker can use xvfb for headed if needed.
    headless = (os.environ.get("AVAILABILITY_HEADLESS") or "true").strip().lower() in ("true", "1", "yes")
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
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
                return
            raise

    raw_text = ""
    raw_html = ""
    summary = "Could not extract availability details from the page."
    log = get_logger()

    log.info("extraction starting url=%s max_seconds=%d", url[:100], max_seconds)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            user_agent=user_agent,
        )
        page = await context.new_page()

        await safe_goto(page, url)
        final_url = page.url
        log.info("extraction page loaded, final_url=%s", final_url[:100] if final_url else "?")
        if time.time() - start > max_seconds:
            log.warning("extraction timeout during load")
            await browser.close()
            return "Timed out while loading the page.", step_index, "", ""

        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception as e:
            log.debug("extraction networkidle wait: %s", e)
        await page.wait_for_timeout(1200)

        if write_screenshot:
            try:
                png = await page.screenshot(full_page=False)
                write_screenshot(step_index, png)
                step_index += 1
            except Exception:
                pass

        # Accept cookies / dismiss overlays (generic - text-based only)
        for selector in [
            'button:has-text("Accept")',
            'button:has-text("I Agree")',
            'button:has-text("Accept All")',
            'button:has-text("OK")',
            '[aria-label*="accept" i]',
        ]:
            try:
                if await page.locator(selector).first.is_visible(timeout=2000):
                    await page.locator(selector).first.click()
                    await page.wait_for_timeout(800)
                    break
            except Exception:
                continue

        # Try to jump to availability section (generic text-based only)
        for selector in [
            'a:has-text("Availability")',
            'a:has-text("Rooms")',
            'button:has-text("See availability")',
            'button:has-text("Check availability")',
            'button:has-text("View rooms")',
        ]:
            try:
                if await page.locator(selector).first.is_visible(timeout=1500):
                    await page.locator(selector).first.click()
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                continue

        # Scroll to content area (generic - look for common patterns)
        try:
            await page.evaluate(
                "() => { "
                "const q = '[class*=\"room\"], [class*=\"Room\"], [id*=\"room\"], [id*=\"availability\"]'; "
                "const el = document.querySelector(q); "
                "if (el) el.scrollIntoView({behavior: 'instant'}); "
                "}"
            )
            await page.wait_for_timeout(1200)
            await page.mouse.wheel(0, 800)
            await page.wait_for_timeout(1000)
        except Exception:
            pass

        # Wait for content to load - Agoda and others load room data via JS
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        # Extra wait for dynamic room content (Agoda, Expedia load async)
        await page.wait_for_timeout(3000)
        # Scroll down to trigger lazy loading
        for _ in range(3):
            await page.mouse.wheel(0, 500)
            await page.wait_for_timeout(800)

        if write_screenshot:
            try:
                png = await page.screenshot(full_page=False)
                write_screenshot(step_index, png)
                step_index += 1
            except Exception:
                pass

        try:
            raw_text = await page.inner_text("body")
        except Exception as e:
            log.warning("extraction inner_text failed: %s", e)
            raw_text = ""
        try:
            raw_html = await page.content()
        except Exception:
            raw_html = ""

        log.info("extraction raw_text len=%d", len(raw_text))
        if raw_text:
            snippet = raw_text[:800].replace("\n", " ").strip()
            log.debug("extraction text snippet: %s...", snippet[:400])
        await browser.close()

    if raw_text:
        lowered = raw_text.lower()
        if "no availability" in lowered or "no rooms available" in lowered or "sold out" in lowered:
            summary = "No availability found for the selected dates."
        else:
            snippet = raw_text[:1500].strip()
            summary = f"Page content (excerpt):\n{snippet}"

    return summary, step_index, raw_text, raw_html
