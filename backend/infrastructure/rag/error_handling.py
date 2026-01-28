"""Error handling utilities for resilient crawling and URL discovery."""
import asyncio
import logging
import time
from typing import Any, Callable, List, Optional, TypeVar, Tuple

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Error types that should trigger retries
RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    asyncio.TimeoutError,
    Exception,  # Catch-all for unexpected errors
)

# Error indicators in responses
BOT_DETECTION_INDICATORS = [
    "cloudflare",
    "captcha",
    "challenge",
    "verify you are human",
    "access denied",
    "bot detected",
    "rate limit",
    "too many requests",
    "blocked",
    "forbidden",
    "403",
    "429",
]

CAPTCHA_INDICATORS = [
    "captcha",
    "recaptcha",
    "hcaptcha",
    "challenge",
    "verify",
    "human verification",
]

AUTH_LOCKOUT_INDICATORS = [
    "authentication required",
    "login required",
    "access denied",
    "unauthorized",
    "401",
    "403",
]


def is_bot_detected(content: str) -> bool:
    """Check if content indicates bot detection."""
    if not content:
        return False
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in BOT_DETECTION_INDICATORS)


def is_captcha_page(content: str) -> bool:
    """Check if content indicates a CAPTCHA page."""
    if not content:
        return False
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in CAPTCHA_INDICATORS)


def is_auth_required(content: str) -> bool:
    """Check if content indicates authentication is required."""
    if not content:
        return False
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in AUTH_LOCKOUT_INDICATORS)


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs,
) -> Optional[T]:
    """
    Retry a function with exponential backoff.
    Returns None if all retries fail.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except RETRYABLE_ERRORS as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after error: {type(e).__name__}: {str(e)[:100]}")
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_retries} retry attempts failed: {type(e).__name__}: {str(e)[:100]}")
        except Exception as e:
            # Non-retryable errors - log and return None
            logger.error(f"Non-retryable error: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    return None


def safe_execute(func: Callable[..., T], default: T, *args, **kwargs) -> T:
    """
    Safely execute a function, returning default value on any error.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Safe execute caught error: {type(e).__name__}: {str(e)[:100]}")
        return default


async def safe_execute_async(func: Callable[..., T], default: T, *args, **kwargs) -> T:
    """
    Safely execute an async function, returning default value on any error.
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Safe execute async caught error: {type(e).__name__}: {str(e)[:100]}")
        return default


def filter_partial_results(results: List[Any], is_valid: Callable[[Any], bool]) -> List[Any]:
    """
    Filter results, keeping only valid ones. Never returns empty unless all were invalid.
    """
    valid = [r for r in results if is_valid(r)]
    return valid if valid else results  # Return original if all invalid (better than empty)


def continue_on_error(func: Callable[..., T], default: T = None, *args, **kwargs) -> T:
    """
    Execute function and continue on error, returning default.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        return default
