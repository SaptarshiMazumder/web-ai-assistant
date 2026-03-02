import asyncio
import html
import logging
import time
import urllib.request
import uuid
import json
import secrets
import os
from datetime import datetime, timezone
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps.auth import get_current_user, require_org_admin, require_super_admin
from api.schemas import (
    AgentConfigPayload,
    AgentConfigResponse,
    AssetCard,
    BotCreateRequest,
    BotCreateResponse,
    BotRenameRequest,
    BotDetailResponse,
    BotDomainAddRequest,
    BotDomainAddResponse,
    BotDomainListResponse,
    BotDomainRecordResponse,
    BotDomainVerifyResponse,
    BotIndexJobListResponse,
    BotIndexJobResponse,
    BotIndexBatchRequest,
    BotIndexRequest,
    BotListResponse,
    BotSourceCreateRequest,
    BotSourceListResponse,
    BotSourceResponse,
    BotSourceSyncSettingsRequest,
    PdfSourceUploadResponse,
    PdfSourceUploadItem,
    TextSourceUploadRequest,
    TextSourceUploadResponse,
    TextSourceUploadItem,
    DocsSourceUploadResponse,
    DocsSourceUploadItem,
    BotSummary,
    Citation,
    ConversationDetailResponse,
    ConversationEndResponse,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationSessionResponse,
    AnalyticsSummaryResponse,
    AnalyticsTimeseriesResponse,
    TopSourcesResponse,
    TopicsResponse,
    RecomputeResponse,
    ConversationSearchResponse,
    ConversationSearchSessionResponse,
    EscalationConfigPayload,
    EscalationConfigResponse,
    EscalationCountsResponse,
    EscalationCreateRequest,
    EscalationListResponse,
    EscalationRecordResponse,
    EscalationStatusUpdateRequest,
    OrgCreateRequest,
    OrgListResponse,
    OrgMemberAddRequest,
    OrgMemberResponse,
    OrgMembersListResponse,
    OrgSelfResponse,
    OrgSummary,
    OrgUpdateRequest,
    TestChatRequest,
    TestChatResponse,
    UrlDiscoveryRequest,
    UrlDiscoveryResponse,
    DiscoveryJobCreateRequest,
    DiscoveryJobResponse,
    DiscoveryJobListResponse,
    BookingLinkJobItem,
    BookingLinkJobsResponse,
    WidgetChatRequest,
    WidgetChatResponse,
    WidgetConfigUpdate,
    PersonaListResponse,
)
from domain.personas import list_personas, list_categories, get_persona, get_persona_system_prompt, get_default_persona_id
from domain.platform_profiles import (
    RESERVATION_PLATFORM_CONFIG,
    ensure_canonical_reservation_url_in_text,
    get_platform_asset_instructions,
    get_platform_features_from_widget,
    get_reservation_config_from_widget,
    get_suggested_messages_for_widget,
)
from application.services.default_prompt_service import (
    build_default_system_instruction,
    extract_business_type_from_widget_config,
)
from application.auth.jwt_auth import is_super_admin
from common.config import config
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from common.di.container import asset_repo, bot_service, conversation_service, indexing_service, org_service, url_discovery, user_service
from common.di.container import analytics_service
from common.logging.chat_debug import chat_debug_emit
from infrastructure.availability.chat_availability import maybe_run_chat_availability

logger = logging.getLogger(__name__)

# Prefixes that indicate availability check failed; don't inject as success.
_AVAILABILITY_ERROR_PREFIXES = ("Could not complete availability check", "The availability check is taking longer")


def _is_real_availability_summary(text: Optional[str]) -> bool:
    """True if we should inject this as availability evidence (not an error message)."""
    if not (text and text.strip()):
        return False
    t = text.strip()
    return not any(t.startswith(prefix) for prefix in _AVAILABILITY_ERROR_PREFIXES)


def _normalize_prompt_text(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _resolve_system_instruction(
    agent_config: dict,
    *,
    lang: str = "en",
    bot_name: str = "",
    widget_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Resolve system instruction:
    1) explicit instructions
    2) explicit non-default persona/custom persona prompt
    3) deterministic default prompt (no LLM)
    """
    default_persona_id = get_default_persona_id()
    instructions = agent_config.get("instructions") if agent_config else None
    if instructions and str(instructions).strip():
        default_persona_prompt = get_persona_system_prompt(default_persona_id, lang=lang) or ""
        if _normalize_prompt_text(instructions) != _normalize_prompt_text(default_persona_prompt):
            return instructions

    persona_id_raw = agent_config.get("persona_id") if agent_config else None
    persona_id = str(persona_id_raw).strip() if persona_id_raw else default_persona_id
    has_explicit_non_default_persona = bool(persona_id_raw and persona_id and persona_id != default_persona_id)

    if has_explicit_non_default_persona:
        # Check built-in personas first.
        builtin = get_persona_system_prompt(persona_id, lang=lang)
        if builtin:
            return builtin

        # Check custom personas stored in agent_config.
        for cp in (agent_config.get("custom_personas") or []):
            if cp.get("id") == persona_id and cp.get("system_prompt"):
                return cp["system_prompt"]

    # Final fallback is deterministic and based on bot name + optional business type.
    business_type = extract_business_type_from_widget_config(widget_config)
    return build_default_system_instruction(
        bot_name=bot_name,
        business_type=business_type,
        lang=lang,
    )


def _get_bot_language(bot) -> str:
    """Extract the bot content language from widget_config. Defaults to 'en'."""
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            wc = json.loads(bot.widget_config)
            lang = (wc.get("language") or "").strip().lower()
            if lang in ("ja", "jp"):
                return "ja"
        except (TypeError, ValueError):
            pass
    return "en"


def _get_language_from_widget_config_dict(widget_config: Optional[Dict[str, Any]]) -> str:
    if isinstance(widget_config, dict):
        lang = str(widget_config.get("language") or "").strip().lower()
        if lang in ("ja", "jp"):
            return "ja"
    return "en"


def _build_deterministic_instruction_for_bot(
    *,
    bot_name: str,
    widget_config: Optional[Dict[str, Any]],
) -> str:
    return build_default_system_instruction(
        bot_name=(bot_name or "").strip(),
        business_type=extract_business_type_from_widget_config(widget_config),
        lang=_get_language_from_widget_config_dict(widget_config),
    )


def _should_autoupdate_to_deterministic_instruction(
    *,
    existing_instructions: str,
    bot_name: str,
    old_widget_config: Optional[Dict[str, Any]],
    new_widget_config: Optional[Dict[str, Any]],
) -> bool:
    """True when instructions are blank/legacy/default and safe to replace."""
    normalized = _normalize_prompt_text(existing_instructions)
    if not normalized:
        return True

    default_persona_id = get_default_persona_id()
    candidates = set()

    for cfg in (old_widget_config or {}, new_widget_config or {}):
        lang = _get_language_from_widget_config_dict(cfg)
        default_persona = get_persona_system_prompt(default_persona_id, lang=lang) or ""
        candidates.add(_normalize_prompt_text(default_persona))
        candidates.add(
            _normalize_prompt_text(
                _build_deterministic_instruction_for_bot(bot_name=bot_name, widget_config=cfg)
            )
        )

    return normalized in candidates


def _initialize_default_agent_config(bot_id: str, display_name: str) -> None:
    """Persist deterministic instructions immediately after bot creation."""
    payload = {
        "persona_id": get_default_persona_id(),
        "instructions": _build_deterministic_instruction_for_bot(
            bot_name=display_name,
            widget_config={},
        ),
    }
    bot_service().update_agent_config(bot_id, json.dumps(payload, ensure_ascii=False))


def _get_booking_url_for_chat(widget_config: Dict[str, Any]) -> Optional[str]:
    """Return a stable booking URL for hotel bots so the model can cite it. None if not hotel or no URL."""
    if widget_config.get("businessType") != "hotel":
        return None
    url = (widget_config.get("bookingTestUrl") or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    if url:
        return url
    pattern = widget_config.get("bookingUrlPattern")
    if isinstance(pattern, dict):
        base = (pattern.get("base_url") or "").strip()
        if base and not base.startswith(("http://", "https://")):
            base = "https://" + base
        if base:
            return base
    return None


def _get_url_bank_for_chat(widget_config: Dict[str, Any], *, limit: int = 20) -> List[Dict[str, str]]:
    """
    Return URL bank entries as {label, url} for injection into chat.
    These are not crawled sources; they're links the assistant can share when relevant.
    """
    raw = widget_config.get("urlBank")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        if url in seen:
            continue
        seen.add(url)
        label = str(item.get("label") or "").strip() or "Link"
        out.append({"label": label, "url": url})
        if len(out) >= limit:
            break
    return out


def _normalize_lang_value(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return "ja" if raw in ("ja", "jp") else "en"


def _normalize_http_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


# ── Reservation (from platform config) ──────────────────────────────────────




from infrastructure.clients.rag_client import run_vertex_rag, run_vertex_rag_stream
from infrastructure.assets.asset_resolver import process_answer_assets, build_asset_instruction, resolve_asset_markers
from infrastructure.services.indexing_service import ensure_bot_corpus
from infrastructure.services.reset_service import delete_gcs_objects, delete_rag_corpora
from infrastructure.db.repositories import PostgresBookingLinkJobRepository, PostgresDiscoveryJobRepository, PostgresIndexJobRepository, PostgresBotRepository
from infrastructure.db.connection import get_connection
from infrastructure.celery_app import celery_app
from infrastructure.tasks.discovery_tasks import discovery_job_task
from domain.entities import DiscoveryJob
from application.services.prompt_generation_service import generate_prompt_from_content

router = APIRouter()


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or str(default)).strip())
    except ValueError:
        return default


# Per-message truncation for conversation context (keeps prompt size bounded).
_CONVERSATION_CONTEXT_MAX_CHARS = max(
    300,
    min(_safe_int_env("CHAT_CONVERSATION_CONTEXT_MAX_CHARS", 1200), 4000),
)


def _format_conversation_context(messages: list) -> str:
    """Format recent messages as 'User: ...' / 'Assistant: ...' with truncation."""
    lines = []
    for m in messages:
        content = (m.content or "").strip()
        if len(content) > _CONVERSATION_CONTEXT_MAX_CHARS:
            content = content[:_CONVERSATION_CONTEXT_MAX_CHARS] + "..."
        role_label = "Assistant" if (m.role or "").lower() == "bot" else "User"
        lines.append(f"{role_label}: {content}")
    return "\n\n".join(lines) if lines else ""


def _require_admin_key(x_admin_key: Optional[str]) -> None:
    required = (getattr(config, "ADMIN_API_KEY", None) or "").strip()
    if not required:
        return
    if (x_admin_key or "").strip() != required:
        raise HTTPException(status_code=401, detail="Missing/invalid admin key")


def _parse_escalation_config(raw: Optional[str]) -> Dict[str, Any]:
    if not raw or not raw.strip():
        return {
            "enabled": False,
            "notify_enabled": False,
            "notify_website": False,
            "notify_instagram": False,
            "notify_line": False,
            "notification_emails": "",
        }
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {
            "enabled": False,
            "notify_enabled": False,
            "notify_website": False,
            "notify_instagram": False,
            "notify_line": False,
            "notification_emails": "",
        }
    legacy = bool(data.get("notify_enabled"))
    return {
        "enabled": bool(data.get("enabled")),
        "notify_enabled": legacy,
        "notify_website": bool(data.get("notify_website")) if "notify_website" in data else legacy,
        "notify_instagram": bool(data.get("notify_instagram")) if "notify_instagram" in data else legacy,
        "notify_line": bool(data.get("notify_line")) if "notify_line" in data else legacy,
        "notification_emails": str(data.get("notification_emails") or ""),
    }


def _require_bot_secret(authorization: Optional[str]) -> str:
    auth = (authorization or "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <bot_secret_key>")
    token = auth.split(" ", 1)[1].strip()
    bot = bot_service().get_bot_by_secret_key(token)
    if not bot:
        raise HTTPException(status_code=401, detail="Invalid bot secret key")
    return bot.bot_id


def _resolve_org_id(user_ctx, org_id: Optional[str]) -> str:
    if org_id:
        if org_id in user_ctx.org_ids or is_super_admin(user_ctx.claims):
            return org_id
        raise HTTPException(status_code=403, detail="Org membership required")
    if len(user_ctx.org_ids) == 1:
        return user_ctx.org_ids[0]
    if is_super_admin(user_ctx.claims):
        raise HTTPException(status_code=400, detail="org_id is required for admin")
    raise HTTPException(status_code=403, detail="Org membership required")


def _assert_bot_org(bot_id: str, org_id: str) -> None:
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    if bot.org_id != org_id:
        raise HTTPException(status_code=403, detail="Bot does not belong to this org")


def _origin_host(origin: Optional[str]) -> str:
    if not origin:
        return ""
    try:
        u = urlparse(origin)
        host = (u.hostname or "").lower()
        return host.split(":")[0]
    except Exception:
        return ""


_rl_window_s = 60
_rl_max_per_window = 60
_rl_state: Dict[str, Tuple[float, int]] = {}


def _rate_limit(bot_id: str) -> None:
    now = time.time()
    start, count = _rl_state.get(bot_id, (now, 0))
    if now - start > _rl_window_s:
        start, count = now, 0
    count += 1
    _rl_state[bot_id] = (start, count)
    if count > _rl_max_per_window:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def _verification_url(hostname: str, token: str) -> str:
    return f"https://{hostname}/.well-known/web-ai-bot-verification.txt"


def _check_domain_verification(hostname: str, token: str, *, timeout_s: float = 5.0) -> Tuple[bool, str]:
    url = _verification_url(hostname, token)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "web-ai-bot-verifier/1.0",
            "Accept": "text/plain,*/*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(64 * 1024).decode("utf-8", errors="ignore")
            if token in body:
                return True, "Verification token found"
            return False, "Well-known file fetched but token was not found in body"
    except Exception as e:
        return False, f"Failed to fetch verification URL: {e}"


@router.post("/v1/bots", response_model=BotCreateResponse)
async def v1_create_bot(
    payload: BotCreateRequest,
    x_admin_key: Optional[str] = Header(default=None),
    org_id: Optional[str] = None,
):
    _require_admin_key(x_admin_key)
    resolved_org_id = (payload.org_id or org_id or "").strip()
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail="org_id is required")
    b = bot_service().create_bot(payload.display_name, resolved_org_id)
    _initialize_default_agent_config(b.bot_id, b.display_name)
    return BotCreateResponse(
        bot_id=b.bot_id,
        display_name=b.display_name,
        publishable_key=b.publishable_key,
        secret_key=b.secret_key,
    )


@router.get("/v1/bots", response_model=BotListResponse)
async def v1_list_bots(x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    bots = bot_service().list_bots()
    return BotListResponse(
        bots=[
            BotSummary(
                bot_id=b.bot_id,
                org_id=b.org_id,
                display_name=b.display_name,
                publishable_key=b.publishable_key,
                secret_key=b.secret_key,
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            for b in bots
        ]
    )


@router.get("/v1/bots/{bot_id}", response_model=BotDetailResponse)
async def v1_get_bot(bot_id: str, x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    return BotDetailResponse(
        bot=BotSummary(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            display_name=bot.display_name,
            publishable_key=bot.publishable_key,
            secret_key=bot.secret_key,
            created_at=bot.created_at,
            updated_at=bot.updated_at,
        )
    )


@router.post("/v1/bots/{bot_id}/domains", response_model=BotDomainAddResponse)
async def v1_add_domain(
    bot_id: str,
    payload: BotDomainAddRequest,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    status, token = bot_service().add_domain(bot_id, payload.hostname)
    hostname = payload.hostname.strip().lower().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    return BotDomainAddResponse(
        bot_id=bot_id,
        hostname=hostname,
        status=status,
        verification_token=token,
        verification_url=_verification_url(hostname, token),
    )


@router.get("/v1/bots/{bot_id}/domains", response_model=BotDomainListResponse)
async def v1_list_domains(bot_id: str, x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    domains = bot_service().list_domains(bot_id)
    return BotDomainListResponse(
        bot_id=bot_id,
        domains=[
            BotDomainRecordResponse(
                org_id=d.org_id,
                bot_id=d.bot_id,
                hostname=d.hostname,
                status=d.status,
                verification_token=d.verification_token,
                verified_at=d.verified_at,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in domains
        ],
    )


@router.post("/v1/bots/{bot_id}/domains/{hostname}/verify", response_model=BotDomainVerifyResponse)
async def v1_verify_domain(
    bot_id: str,
    hostname: str,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")

    status, token = bot_service().add_domain(bot_id, hostname)
    ok, msg = _check_domain_verification(hostname, token)
    if ok:
        bot_service().mark_domain_verified(bot_id, hostname)
        return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status="verified", verified=True, message=msg)
    return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status=status, verified=False, message=msg)


@router.post("/v1/bots/{bot_id}/index")
async def v1_start_index(
    bot_id: str,
    payload: BotIndexRequest,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")

    try:
        return await indexing_service().start_indexing_for_bot(bot_id, payload.url, headless=payload.headless)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/bots/{bot_id}/jobs", response_model=BotIndexJobListResponse)
async def v1_list_jobs(bot_id: str, x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    jobs = indexing_service().list_jobs_for_bot(bot_id)
    return BotIndexJobListResponse(
        bot_id=bot_id,
        jobs=[
            BotIndexJobResponse(
                job_id=j.job_id,
                url=j.url,
                hostname=j.hostname,
                stage=j.stage,
                pages_crawled=j.pages_crawled,
                docs_count=j.docs_count,
                gcs_prefix=j.gcs_prefix,
                last_error=j.last_error,
                created_at=j.created_at,
                updated_at=j.updated_at,
                crawled_urls=getattr(j, "crawled_urls", None) or [],
                source_id=getattr(j, "source_id", None),
            )
            for j in jobs
        ],
    )


@router.get("/v1/bots/{bot_id}/index/status")
async def v1_index_status(
    bot_id: str,
    url: str,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    try:
        return indexing_service().get_job_status_by_hostname(bot_id, url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/bots/{bot_id}/index/cancel")
async def v1_cancel_index(
    bot_id: str,
    payload: BotIndexRequest,
    authorization: Optional[str] = Header(default=None),
):
    owner_bot_id = _require_bot_secret(authorization)
    if owner_bot_id != bot_id:
        raise HTTPException(status_code=403, detail="Bot secret does not match bot_id")
    try:
        return indexing_service().cancel_job(bot_id, payload.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/pk/{publishable_key}/index")
async def v1_pk_start_index(
    publishable_key: str,
    payload: BotIndexRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    _rate_limit(bot.bot_id)
    try:
        return await indexing_service().start_indexing_for_bot(bot.bot_id, payload.url, headless=payload.headless)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/pk/{publishable_key}/index/status")
async def v1_pk_index_status(
    publishable_key: str,
    url: str,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    try:
        return indexing_service().get_job_status_by_hostname(bot.bot_id, url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/pk/{publishable_key}/index/cancel")
async def v1_pk_cancel_index(
    publishable_key: str,
    payload: BotIndexRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    try:
        return indexing_service().cancel_job(bot.bot_id, payload.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v1/pk/{publishable_key}/widget-config")
async def v1_pk_widget_config(publishable_key: str):
    """Public: return saved widget config for the bot. Used by the embed script on load."""
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    
    # Parse widget config or start with empty dict
    config = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            config = {}
    
    # Fallback: if no custom title is set, use the bot's display_name
    if not config.get("title"):
        config["title"] = getattr(bot, "display_name", "Chat")
    
    # Inject suggested messages from platform profile or default (config-driven, all channels)
    lang = _get_language_from_widget_config_dict(config)
    config["suggestedMessages"] = get_suggested_messages_for_widget(config, lang=lang)
    config["suggestedMessagesEnabled"] = True
    
    return config


@router.get("/v1/pk/{publishable_key}/menu", response_class=HTMLResponse)
async def v1_pk_menu_page(publishable_key: str, request: Request):
    """Public: scrollable menu page for the bot. Used when user taps menu on Instagram."""
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    items = asset_repo().list_assets_for_bot(bot.bot_id, active_only=True, asset_type="menu_item")
    base = str(request.base_url).rstrip("/")
    title = getattr(bot, "display_name", "Menu") or "Menu"
    _MENU_CATEGORY_ORDER = ("course", "dish", "drink", "lunch", "menu")
    _MENU_CATEGORY_LABELS = {
        "course": "Party / Course",
        "dish": "Dish",
        "drink": "Drink",
        "lunch": "Lunch",
        "menu": "Menu",
    }
    grouped: Dict[str, List[Any]] = {k: [] for k in _MENU_CATEGORY_ORDER}
    for item in items:
        meta = item.metadata if isinstance(getattr(item, "metadata", None), dict) else {}
        raw = str(meta.get("category") or "").strip().lower()
        aliases = {"party": "course", "plan": "course", "set": "course", "beverage": "drink", "food": "dish", "lunch_set": "lunch"}
        cat = aliases.get(raw, raw) if raw else "menu"
        cat = cat if cat in _MENU_CATEGORY_ORDER else "menu"
        grouped[cat].append(item)
    html_sections: List[str] = []
    for cat in _MENU_CATEGORY_ORDER:
        cat_items = grouped.get(cat) or []
        if not cat_items:
            continue
        label = _MENU_CATEGORY_LABELS.get(cat, cat.title())
        section_items: List[str] = []
        for a in cat_items:
            name = str(a.name or "Menu item").strip()
            meta = a.metadata if isinstance(getattr(a, "metadata", None), dict) else {}
            price_text = str(meta.get("price_text") or "").strip()
            if not price_text and isinstance(meta.get("price"), dict):
                price_text = str((meta.get("price") or {}).get("text") or "").strip()
            details = str(meta.get("details") or "").strip() or (a.description or "").strip()
            img_url = (a.image_public_url or "").strip()
            if img_url and not img_url.startswith(("http://", "https://")):
                img_url = f"{base}{img_url}" if img_url.startswith("/") else f"{base}/{img_url}"
            img_html = f'<img src="{html.escape(img_url)}" alt="" loading="lazy" />' if img_url else ""
            price_html = f'<span class="price">{html.escape(price_text)}</span>' if price_text else ""
            details_html = f'<p class="details">{html.escape(details)}</p>' if details else ""
            section_items.append(
                f'<li class="menu-item">'
                f'<div class="item-img">{img_html}</div>'
                f'<div class="item-info"><h4>{html.escape(name)}</h4>{price_html}{details_html}</div>'
                f'</li>'
            )
        html_sections.append(
            f'<section class="category"><h2>{html.escape(label)}</h2><ul>{"".join(section_items)}</ul></section>'
        )
    body = "\n".join(html_sections) if html_sections else '<p class="empty">No menu items available.</p>'
    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - Menu</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px; background: #f5f5f5; color: #222; }}
h1 {{ font-size: 1.25rem; margin: 0 0 16px; }}
h2 {{ font-size: 1rem; margin: 16px 0 8px; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }}
.category {{ margin-bottom: 24px; }}
.menu-item {{ display: flex; gap: 12px; margin-bottom: 16px; padding: 12px; background: #fff; border-radius: 8px; list-style: none; }}
.item-img {{ flex-shrink: 0; width: 80px; height: 80px; border-radius: 6px; overflow: hidden; background: #eee; }}
.item-img img {{ width: 100%; height: 100%; object-fit: cover; }}
.item-info {{ flex: 1; min-width: 0; }}
.item-info h4 {{ margin: 0 0 4px; font-size: 0.95rem; }}
.price {{ color: #c00; font-weight: 600; }}
.details {{ margin: 4px 0 0; font-size: 0.85rem; color: #666; }}
.empty {{ color: #666; }}
</style>
</head>
<body>
<h1>{html.escape(title)} - Menu</h1>
{body}
</body>
</html>"""
    return HTMLResponse(content=page_html)


@router.get("/v1/pk/{publishable_key}/escalation-config", response_model=EscalationConfigResponse)
async def v1_pk_escalation_config(publishable_key: str):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    cfg = _parse_escalation_config(getattr(bot, "escalation_config", None))
    return EscalationConfigResponse(
        enabled=cfg["enabled"],
        notify_enabled=cfg["notify_enabled"],
        notify_website=cfg["notify_website"],
        notify_instagram=cfg["notify_instagram"],
        notify_line=cfg["notify_line"],
        notification_emails=cfg["notification_emails"],
    )


@router.post("/v1/pk/{publishable_key}/chat", response_model=WidgetChatResponse)
async def v1_widget_chat(
    publishable_key: str,
    payload: WidgetChatRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    trace_id = uuid.uuid4().hex
    chat_debug_emit(
        {
            "type": "chat_request_received",
            "trace_id": trace_id,
            "publishable_key": publishable_key,
            "payload": payload.model_dump() if hasattr(payload, "model_dump") else getattr(payload, "__dict__", {}),
        }
    )
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Unknown bot publishable key"})
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")

    if config.REQUIRE_DOMAIN_VERIFICATION:
        origin_h = _origin_host(origin)
        if origin_h:
            verified_hosts = set(bot_service().list_verified_hosts(bot.bot_id))
            if origin_h not in verified_hosts:
                raise HTTPException(status_code=403, detail="Origin not allowed for this bot")

    _rate_limit(bot.bot_id)

    corpus = ensure_bot_corpus(bot.bot_id)

    msg = (payload.message or "").strip()
    if not msg:
        chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Missing message"})
        raise HTTPException(status_code=400, detail="Missing message")

    site_url = (getattr(payload, "site_url", None) or "").strip()
    site_title = (getattr(payload, "site_title", None) or "").strip()
    ctx_lines = []
    if site_title:
        ctx_lines.append(f"Site title: {site_title}")
    if site_url:
        ctx_lines.append(f"Site URL: {site_url}")
    query = msg
    if ctx_lines:
        query = f"{msg}\n\nContext:\n" + "\n".join(ctx_lines)

    allowed_host = ""
    if site_url:
        try:
            hostname = (urlparse(site_url).hostname or "").lower().split(":")[0]
            if hostname in ("localhost", "127.0.0.1"):
                allowed_host = ""  # Dashboard Testing tab: skip host filter so all RAG evidence is used
            else:
                allowed_host = hostname
        except Exception:
            allowed_host = ""

    chat_debug_emit(
        {
            "type": "chat_context",
            "trace_id": trace_id,
            "bot_id": bot.bot_id,
            "corpus_resource": corpus,
            "allowed_host": allowed_host,
            "site_url": site_url,
            "site_title": site_title,
            "raw_message": msg,
            "final_query": query,
        }
    )

    def _rag_dbg(evt: Dict[str, Any]) -> None:
        evt2 = dict(evt)
        evt2["trace_id"] = trace_id
        chat_debug_emit(evt2)

    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass

    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = _resolve_system_instruction(
        agent_config,
        lang=_get_bot_language(bot),
        bot_name=getattr(bot, "display_name", "") or "",
        widget_config=widget_config,
    )
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    session = conversation_service().get_or_create_session(
        bot_id=bot.bot_id,
        org_id=bot.org_id,
        channel="chat",
        session_id=getattr(payload, "session_id", None),
        site_url=site_url or None,
        site_title=site_title or None,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=msg,
    )
    recent = conversation_service().list_recent_messages(session.session_id, limit=CONVERSATION_HISTORY_MESSAGES)
    conversation_context = _format_conversation_context(recent)

    # If message asks about availability and bot has booking config, run sync check and inject as evidence
    extra_evidence: List[Dict[str, str]] = []
    availability_summary, availability_skip_reason = maybe_run_chat_availability(bot.bot_id, msg, widget_config)
    if availability_skip_reason:
        chat_debug_emit({"type": "chat_availability_skipped", "trace_id": trace_id, "reason": availability_skip_reason})
    if availability_summary and _is_real_availability_summary(availability_summary):
        extra_evidence.append({"url": "Live availability check", "snippet": availability_summary})
        chat_debug_emit({"type": "chat_availability_injected", "trace_id": trace_id})
    booking_url = _get_booking_url_for_chat(widget_config)
    if booking_url:
        extra_evidence.append({"url": booking_url, "snippet": f"To book or check availability, visit: {booking_url}"})

    url_bank = _get_url_bank_for_chat(widget_config)
    if url_bank:
        for it in url_bank:
            label = it["label"]
            url = it["url"]
            extra_evidence.append(
                {
                    "url": url,
                    "snippet": f"If the customer asks about {label}, share this link: [{label}]({url})",
                }
            )
        bank_lines = "\n".join([f"- {it['label']}: [{it['label']}]({it['url']})" for it in url_bank])
        bank_instruction = (
            "Answer links (use only when they match the customer’s question):\n"
            f"{bank_lines}\n\n"
            "If the question matches one of these topics, answer from that URL's content when it appears in the evidence, and include the matching link in your response.\n"
            "Write links as markdown like [Pricing](https://...) inside a normal sentence."
        )
        system_instruction = f"{system_instruction}\n\n{bank_instruction}" if system_instruction else bank_instruction

    # Inject reservation config from platform profile if applicable
    reservation_cfg = get_reservation_config_from_widget(widget_config, lang=_get_bot_language(bot))
    if reservation_cfg:
        extra_evidence.append({
            "url": reservation_cfg["url"],
            "snippet": f"Official online reservation page: {reservation_cfg['url']}",
        })
        system_instruction = (
            f"{system_instruction}\n\n{reservation_cfg['instruction']}"
            if system_instruction
            else reservation_cfg["instruction"]
        )

    # Inject asset bank as system instruction (up to 150 items; URLs resolved server-side)
    asset_instruction = build_asset_instruction(bot.bot_id)
    if asset_instruction:
        system_instruction = f"{system_instruction}\n\n{asset_instruction}" if system_instruction else asset_instruction
    platform_asset_instruction = get_platform_asset_instructions(widget_config, lang=_get_bot_language(bot))
    if platform_asset_instruction:
        system_instruction = f"{system_instruction}\n\n{platform_asset_instruction}" if system_instruction else platform_asset_instruction

    result = run_vertex_rag(
        query,
        rag_corpus=corpus,
        allowed_host=None,
        debug_cb=_rag_dbg,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        conversation_context=conversation_context or None,
        extra_evidence=extra_evidence if extra_evidence else None,
        bot_display_name=getattr(bot, "display_name", None),
    )
    chat_debug_emit({"type": "chat_rag_result", "trace_id": trace_id, "result": result})
    sources = result.get("sources") or []
    citations = []
    for s in sources:
        citations.append(Citation(url=str(s.get("url") or ""), snippet=str(s.get("excerpt") or "")))

    if not citations:
        host_label = allowed_host or "this site"
        chat_debug_emit(
            {
                "type": "chat_refusal",
                "trace_id": trace_id,
                "reason": "no_citations_after_host_filter",
                "host_label": host_label,
            }
        )
        answer = f"I can’t find that in the information I’ve learned for {host_label} yet. Try asking about something else, or add more sources in the dashboard."
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="bot",
            content=answer,
            citations=[],
        )
        return WidgetChatResponse(
            answer=answer,
            citations=[],
            session_id=session.session_id,
        )

    chat_debug_emit(
        {
            "type": "chat_response",
            "trace_id": trace_id,
            "answer": str(result.get("answer") or ""),
            "citations": [c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
        }
    )
    answer = str(result.get("answer") or "")

    # Ensure reservation URLs use the canonical one from config
    if reservation_cfg:
        answer = ensure_canonical_reservation_url_in_text(
            answer, reservation_cfg["url"], reservation_cfg["domain_key"]
        )

    platform_features = get_platform_features_from_widget(widget_config)
    menu_extraction_enabled = platform_features.get("menu_extraction_enabled") if platform_features else False
    allowed_asset_types = {"menu_item"} if menu_extraction_enabled else None

    # Extract {{asset:ID}} markers from the LLM answer first
    answer, marker_cards = resolve_asset_markers(
        answer, bot.bot_id, session.session_id, allowed_asset_types=allowed_asset_types
    )
    if marker_cards:
        asset_cards = marker_cards
    else:
        # Fallback to keyword-based matching
        answer, asset_cards = process_answer_assets(
            answer,
            bot.bot_id,
            user_query=msg,
            session_id=session.session_id,
            allowed_asset_types=allowed_asset_types,
        )
    assets = [AssetCard(**c) for c in asset_cards]

    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
        citations=[c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
    )
    return WidgetChatResponse(answer=answer, citations=citations, assets=assets, session_id=session.session_id)


@router.post("/v1/pk/{publishable_key}/chat/stream")
async def v1_widget_chat_stream(
    publishable_key: str,
    payload: WidgetChatRequest,
    request: Request,
    origin: Optional[str] = Header(default=None),
):
    trace_id = uuid.uuid4().hex
    chat_debug_emit(
        {
            "type": "chat_request_received",
            "trace_id": trace_id,
            "publishable_key": publishable_key,
            "payload": payload.model_dump() if hasattr(payload, "model_dump") else getattr(payload, "__dict__", {}),
        }
    )
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Unknown bot publishable key"})
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")

    if config.REQUIRE_DOMAIN_VERIFICATION:
        origin_h = _origin_host(origin)
        if origin_h:
            verified_hosts = set(bot_service().list_verified_hosts(bot.bot_id))
            if origin_h not in verified_hosts:
                raise HTTPException(status_code=403, detail="Origin not allowed for this bot")

    _rate_limit(bot.bot_id)

    corpus = ensure_bot_corpus(bot.bot_id)

    msg = (payload.message or "").strip()
    if not msg:
        chat_debug_emit({"type": "chat_error", "trace_id": trace_id, "error": "Missing message"})
        raise HTTPException(status_code=400, detail="Missing message")

    site_url = (getattr(payload, "site_url", None) or "").strip()
    site_title = (getattr(payload, "site_title", None) or "").strip()
    ctx_lines = []
    if site_title:
        ctx_lines.append(f"Site title: {site_title}")
    if site_url:
        ctx_lines.append(f"Site URL: {site_url}")
    query = msg
    if ctx_lines:
        query = f"{msg}\n\nContext:\n" + "\n".join(ctx_lines)

    allowed_host = ""
    if site_url:
        try:
            hostname = (urlparse(site_url).hostname or "").lower().split(":")[0]
            if hostname in ("localhost", "127.0.0.1"):
                allowed_host = ""  # Dashboard Testing tab: skip host filter so all RAG evidence is used
            else:
                allowed_host = hostname
        except Exception:
            allowed_host = ""

    chat_debug_emit(
        {
            "type": "chat_context",
            "trace_id": trace_id,
            "bot_id": bot.bot_id,
            "corpus_resource": corpus,
            "allowed_host": allowed_host,
            "site_url": site_url,
            "site_title": site_title,
            "raw_message": msg,
            "final_query": query,
        }
    )

    def _rag_dbg(evt: Dict[str, Any]) -> None:
        evt2 = dict(evt)
        evt2["trace_id"] = trace_id
        chat_debug_emit(evt2)

    widget_config_stream: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config_stream = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass

    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = _resolve_system_instruction(
        agent_config,
        lang=_get_bot_language(bot),
        bot_name=getattr(bot, "display_name", "") or "",
        widget_config=widget_config_stream,
    )
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    session = conversation_service().get_or_create_session(
        bot_id=bot.bot_id,
        org_id=bot.org_id,
        channel="chat",
        session_id=getattr(payload, "session_id", None),
        site_url=site_url or None,
        site_title=site_title or None,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=msg,
    )
    recent = conversation_service().list_recent_messages(session.session_id, limit=CONVERSATION_HISTORY_MESSAGES)
    conversation_context = _format_conversation_context(recent)

    # If message asks about availability and bot has booking config, run sync check and inject as evidence
    extra_evidence_stream: List[Dict[str, str]] = []
    availability_summary_stream, availability_skip_reason_stream = maybe_run_chat_availability(bot.bot_id, msg, widget_config_stream)
    if availability_skip_reason_stream:
        chat_debug_emit({"type": "chat_availability_skipped", "trace_id": trace_id, "reason": availability_skip_reason_stream})
    if availability_summary_stream and _is_real_availability_summary(availability_summary_stream):
        extra_evidence_stream.append({"url": "Live availability check", "snippet": availability_summary_stream})
        chat_debug_emit({"type": "chat_availability_injected", "trace_id": trace_id})
    booking_url_stream = _get_booking_url_for_chat(widget_config_stream)
    if booking_url_stream:
        extra_evidence_stream.append({"url": booking_url_stream, "snippet": f"To book or check availability, visit: {booking_url_stream}"})

    url_bank_stream = _get_url_bank_for_chat(widget_config_stream)
    if url_bank_stream:
        for it in url_bank_stream:
            label = it["label"]
            url = it["url"]
            extra_evidence_stream.append(
                {
                    "url": url,
                    "snippet": f"If the customer asks about {label}, share this link: [{label}]({url})",
                }
            )
        bank_lines_stream = "\n".join([f"- {it['label']}: [{it['label']}]({it['url']})" for it in url_bank_stream])
        bank_instruction_stream = (
            "Answer links (use only when they match the customer’s question):\n"
            f"{bank_lines_stream}\n\n"
            "If the question matches one of these topics, answer from that URL's content when it appears in the evidence, and include the matching link in your response.\n"
            "Write links as markdown like [Pricing](https://...) inside a normal sentence."
        )
        system_instruction = f"{system_instruction}\n\n{bank_instruction_stream}" if system_instruction else bank_instruction_stream

    # Inject restaurant reservation instructions if applicable
    reservation_cfg_stream = get_reservation_config_from_widget(widget_config_stream, lang=_get_bot_language(bot))
    if reservation_cfg_stream:
        extra_evidence_stream.append({
            "url": reservation_cfg_stream["url"],
            "snippet": f"Official online reservation page: {reservation_cfg_stream['url']}",
        })
        system_instruction = (
            f"{system_instruction}\n\n{reservation_cfg_stream['instruction']}"
            if system_instruction
            else reservation_cfg_stream["instruction"]
        )

    # Inject asset bank as system instruction (up to 150 items; URLs resolved server-side)
    asset_instruction_stream = build_asset_instruction(bot.bot_id)
    if asset_instruction_stream:
        system_instruction = f"{system_instruction}\n\n{asset_instruction_stream}" if system_instruction else asset_instruction_stream
    platform_asset_instruction_stream = get_platform_asset_instructions(widget_config_stream, lang=_get_bot_language(bot))
    if platform_asset_instruction_stream:
        system_instruction = f"{system_instruction}\n\n{platform_asset_instruction_stream}" if system_instruction else platform_asset_instruction_stream

    async def _gen():
        yield json.dumps({"type": "meta", "session_id": session.session_id}, ensure_ascii=False) + "\n"
        try:
            for evt in run_vertex_rag_stream(
                query,
                rag_corpus=corpus,
                allowed_host=None,
                debug_cb=_rag_dbg,
                system_instruction=system_instruction,
                model_name=model_name,
                temperature=temperature,
                conversation_context=conversation_context or None,
                extra_evidence=extra_evidence_stream if extra_evidence_stream else None,
                bot_display_name=getattr(bot, "display_name", None),
            ):
                if evt.get("type") == "delta":
                    yield json.dumps({"type": "delta", "text": evt.get("text") or ""}, ensure_ascii=False) + "\n"
                    continue
                if evt.get("type") == "done":
                    sources = evt.get("sources") or []
                    citations = []
                    for s in sources:
                        citations.append({"url": str(s.get("url") or ""), "snippet": str(s.get("excerpt") or "")})
                    if not citations:
                        host_label = allowed_host or "this site"
                        chat_debug_emit(
                            {
                                "type": "chat_refusal",
                                "trace_id": trace_id,
                                "reason": "no_citations_after_host_filter",
                                "host_label": host_label,
                            }
                        )
                        answer = f"I can?t find that in the indexed content for {host_label}. Try asking about something on the site, or re-run Crawl."
                        conversation_service().add_message(
                            session_id=session.session_id,
                            bot_id=bot.bot_id,
                            role="bot",
                            content=answer,
                            citations=[],
                        )
                        yield json.dumps(
                            {
                                "type": "done",
                                "answer": answer,
                                "citations": [],
                                "session_id": session.session_id,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    else:
                        answer = str(evt.get("answer") or "")
                        if reservation_cfg_stream:
                            answer = ensure_canonical_reservation_url_in_text(
                                answer, reservation_cfg_stream["url"], reservation_cfg_stream["domain_key"]
                            )
                        platform_features_stream = get_platform_features_from_widget(widget_config_stream)
                        menu_extraction_enabled_stream = platform_features_stream.get("menu_extraction_enabled") if platform_features_stream else False
                        allowed_asset_types_stream = {"menu_item"} if menu_extraction_enabled_stream else None
                        # Extract {{asset:ID}} markers from the LLM answer first
                        answer, marker_cards_stream = resolve_asset_markers(
                            answer, bot.bot_id, session.session_id, allowed_asset_types=allowed_asset_types_stream
                        )
                        if marker_cards_stream:
                            asset_cards_stream = marker_cards_stream
                        else:
                            # Fallback to keyword-based matching
                            answer, asset_cards_stream = process_answer_assets(
                                answer,
                                bot.bot_id,
                                user_query=msg,
                                session_id=session.session_id,
                                allowed_asset_types=allowed_asset_types_stream,
                            )
                        chat_debug_emit(
                            {
                                "type": "chat_response",
                                "trace_id": trace_id,
                                "answer": answer,
                                "citations": citations,
                                "assets": asset_cards_stream,
                            }
                        )
                        conversation_service().add_message(
                            session_id=session.session_id,
                            bot_id=bot.bot_id,
                            role="bot",
                            content=answer,
                            citations=citations,
                        )
                        yield json.dumps(
                            {
                                "type": "done",
                                "answer": answer,
                                "citations": citations,
                                "assets": asset_cards_stream,
                                "session_id": session.session_id,
                            },
                            ensure_ascii=False,
                        ) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        _gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": session.session_id,
        },
    )


@router.get("/v1/admin/orgs", response_model=OrgListResponse)
async def v1_admin_list_orgs(user=Depends(require_super_admin)):
    orgs = org_service().list_orgs()
    return OrgListResponse(
        orgs=[
            OrgSummary(
                org_id=o.org_id,
                name=o.name,
                status=o.status,
                plan=o.plan,
                stripe_customer_id=o.stripe_customer_id,
                stripe_subscription_id=o.stripe_subscription_id,
                created_at=o.created_at,
                updated_at=o.updated_at,
            )
            for o in orgs
        ],
    )


@router.post("/v1/admin/orgs", response_model=OrgSummary)
async def v1_admin_create_org(payload: OrgCreateRequest, user=Depends(require_super_admin)):
    org_id = org_service().create_org(payload.name)
    orgs = org_service().list_orgs()
    org = next((o for o in orgs if o.org_id == org_id), None)
    if not org:
        raise HTTPException(status_code=500, detail="Failed to create org")
    return OrgSummary(
        org_id=org.org_id,
        name=org.name,
        status=org.status,
        plan=org.plan,
        stripe_customer_id=org.stripe_customer_id,
        stripe_subscription_id=org.stripe_subscription_id,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.post("/v1/admin/orgs/{org_id}/disable")
async def v1_admin_disable_org(org_id: str, user=Depends(require_super_admin)):
    org_service().set_org_status(org_id, "disabled")
    return {"status": "ok"}


@router.post("/v1/admin/orgs/{org_id}/enable")
async def v1_admin_enable_org(org_id: str, user=Depends(require_super_admin)):
    org_service().set_org_status(org_id, "active")
    return {"status": "ok"}


@router.get("/v1/admin/orgs/{org_id}/members", response_model=OrgMembersListResponse)
async def v1_admin_list_org_members(org_id: str, user=Depends(require_super_admin)):
    members = org_service().list_org_members(org_id)
    return OrgMembersListResponse(
        org_id=org_id,
        members=[
            OrgMemberResponse(
                user_id=m.user_id,
                email=m.email,
                first_name=m.first_name,
                last_name=m.last_name,
                role=m.role,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in members
        ],
    )


@router.post("/v1/admin/orgs/{org_id}/members")
async def v1_admin_add_org_member(org_id: str, payload: OrgMemberAddRequest, user=Depends(require_super_admin)):
    placeholder = user_service().create_user_placeholder(payload.email)
    org_service().add_membership(org_id, placeholder.user_id, payload.role)
    return {"status": "ok"}


@router.get("/v1/org/self", response_model=OrgSelfResponse)
async def v1_org_self(user=Depends(get_current_user)):
    return OrgSelfResponse(org_ids=user.org_ids)


@router.get("/v1/org/info", response_model=OrgSummary)
async def v1_org_info(org_id: str, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    org = org_service().get_org(resolved_org)
    if not org:
        raise HTTPException(status_code=404, detail="Unknown org_id")
    return OrgSummary(
        org_id=org.org_id,
        name=org.name,
        status=org.status,
        plan=org.plan,
        stripe_customer_id=org.stripe_customer_id,
        stripe_subscription_id=org.stripe_subscription_id,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.post("/v1/org/name", response_model=OrgSummary)
async def v1_org_update_name(org_id: str, payload: OrgUpdateRequest, user=Depends(require_org_admin)):
    resolved_org = _resolve_org_id(user, org_id)
    org_service().update_org_name(resolved_org, payload.name)
    org = org_service().get_org(resolved_org)
    if not org:
        raise HTTPException(status_code=404, detail="Unknown org_id")
    return OrgSummary(
        org_id=org.org_id,
        name=org.name,
        status=org.status,
        plan=org.plan,
        stripe_customer_id=org.stripe_customer_id,
        stripe_subscription_id=org.stripe_subscription_id,
        created_at=org.created_at,
        updated_at=org.updated_at,
    )


@router.get("/v1/org/members", response_model=OrgMembersListResponse)
async def v1_org_list_members(org_id: str, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    members = org_service().list_org_members(resolved_org)
    return OrgMembersListResponse(
        org_id=resolved_org,
        members=[
            OrgMemberResponse(
                user_id=m.user_id,
                email=m.email,
                first_name=m.first_name,
                last_name=m.last_name,
                role=m.role,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in members
        ],
    )


@router.post("/v1/org/members")
async def v1_org_add_member(org_id: str, payload: OrgMemberAddRequest, user=Depends(require_org_admin)):
    resolved_org = _resolve_org_id(user, org_id)
    if not is_super_admin(user.claims):
        memberships = org_service().get_org_memberships(user.user_id)
        role = next((m["role"] for m in memberships if m["org_id"] == resolved_org), "")
        if role not in {"org_admin", "owner"}:
            raise HTTPException(status_code=403, detail="Org admin access required")
    placeholder = user_service().create_user_placeholder(payload.email)
    org_service().add_membership(resolved_org, placeholder.user_id, payload.role)
    return {"status": "ok"}


@router.get("/v1/org/bots", response_model=BotListResponse)
async def v1_org_list_bots(org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    bots = bot_service().list_bots(resolved_org)
    return BotListResponse(
        bots=[
            BotSummary(
                bot_id=b.bot_id,
                org_id=b.org_id,
                display_name=b.display_name,
                publishable_key=b.publishable_key,
                secret_key=b.secret_key,
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            for b in bots
        ]
    )


@router.post("/v1/org/bots", response_model=BotCreateResponse)
async def v1_org_create_bot(payload: BotCreateRequest, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    b = bot_service().create_bot(payload.display_name, resolved_org)
    _initialize_default_agent_config(b.bot_id, b.display_name)
    return BotCreateResponse(
        bot_id=b.bot_id,
        display_name=b.display_name,
        publishable_key=b.publishable_key,
        secret_key=b.secret_key,
    )


@router.get("/v1/org/bots/{bot_id}", response_model=BotDetailResponse)
async def v1_org_get_bot(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    widget_config: Optional[Dict[str, Any]] = None
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    # When suggestedMessages is empty, merge platform config so dashboard shows Menu, Reservation, etc.
    if isinstance(widget_config, dict):
        wc_suggested = widget_config.get("suggestedMessages")
        if not (isinstance(wc_suggested, list) and wc_suggested):
            lang = _get_language_from_widget_config_dict(widget_config)
            resolved = get_suggested_messages_for_widget(widget_config, lang=lang)
            if resolved:
                # Convert to dashboard format: id, label, type, prompt
                widget_config = dict(widget_config)
                widget_config["suggestedMessages"] = [
                    {"id": m["id"], "label": m["label"], "type": m["type"], "prompt": m.get("prompt") or m["label"]}
                    for m in resolved
                ]
    return BotDetailResponse(
        bot=BotSummary(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            display_name=bot.display_name,
            publishable_key=bot.publishable_key,
            secret_key=bot.secret_key,
            created_at=bot.created_at,
            updated_at=bot.updated_at,
        ),
        widget_config=widget_config,
    )


@router.patch("/v1/org/bots/{bot_id}", response_model=BotSummary)
async def v1_org_rename_bot(
    bot_id: str,
    payload: BotRenameRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    new_name = (payload.display_name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="display_name is required")
    if len(new_name) > 120:
        raise HTTPException(status_code=400, detail="display_name is too long")
    bot_service().update_display_name(bot_id, new_name)
    updated = bot_service().get_bot_record(bot_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    return BotSummary(
        bot_id=updated.bot_id,
        org_id=updated.org_id,
        display_name=updated.display_name,
        publishable_key=updated.publishable_key,
        secret_key=updated.secret_key,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
    )


@router.put("/v1/org/bots/{bot_id}/widget-config")
async def v1_org_update_bot_widget_config(
    bot_id: str,
    payload: WidgetConfigUpdate,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    incoming = payload.model_dump(exclude_none=True)

    # Merge with existing widget_config so partial updates don't wipe unrelated keys
    # (e.g., urlBank/answer links).
    existing: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            existing = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    merged = {**existing, **incoming}
    config_json = json.dumps(merged)
    bot_service().update_widget_config(bot_id, config_json)

    # Keep deterministic default instructions in sync with businessType/language,
    # but never overwrite explicit custom instructions or explicit non-default persona.
    agent_config: Dict[str, Any] = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            agent_config = {}

    default_persona_id = get_default_persona_id()
    persona_id = str(agent_config.get("persona_id") or "").strip()
    is_explicit_non_default_persona = bool(persona_id and persona_id != default_persona_id)

    existing_instructions = str(agent_config.get("instructions") or "")
    if (
        not is_explicit_non_default_persona
        and _should_autoupdate_to_deterministic_instruction(
            existing_instructions=existing_instructions,
            bot_name=(bot.display_name or "").strip(),
            old_widget_config=existing,
            new_widget_config=merged,
        )
    ):
        agent_config["instructions"] = _build_deterministic_instruction_for_bot(
            bot_name=(bot.display_name or "").strip(),
            widget_config=merged,
        )
        if not str(agent_config.get("persona_id") or "").strip():
            agent_config["persona_id"] = default_persona_id
        bot_service().update_agent_config(bot_id, json.dumps(agent_config, ensure_ascii=False))

    return {"status": "ok", "bot_id": bot_id}


@router.get("/v1/org/bots/{bot_id}/agent-config", response_model=AgentConfigResponse)
async def v1_org_get_agent_config(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    custom_personas = agent_config.get("custom_personas") or []
    raw_persona_id = str(agent_config.get("persona_id") or "").strip()
    custom_ids = {
        str(cp.get("id") or "").strip()
        for cp in custom_personas
        if isinstance(cp, dict) and cp.get("id")
    }
    if raw_persona_id and (get_persona_system_prompt(raw_persona_id) or raw_persona_id in custom_ids):
        effective_persona_id = raw_persona_id
    else:
        effective_persona_id = get_default_persona_id()
    bot_lang = _get_bot_language(bot)
    instructions = agent_config.get("instructions")
    default_persona_prompt = get_persona_system_prompt(get_default_persona_id(), lang=bot_lang) or ""
    if isinstance(instructions, str) and _normalize_prompt_text(instructions) == _normalize_prompt_text(default_persona_prompt):
        instructions = ""
    if not (isinstance(instructions, str) and instructions.strip()):
        widget_config: Dict[str, Any] = {}
        if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
            try:
                widget_config = json.loads(bot.widget_config)
            except (TypeError, ValueError):
                widget_config = {}
        instructions = build_default_system_instruction(
            bot_name=(bot.display_name or "").strip(),
            business_type=extract_business_type_from_widget_config(widget_config),
            lang=bot_lang,
        )
    return AgentConfigResponse(
        model_id=agent_config.get("model_id"),
        instructions=instructions,
        temperature=agent_config.get("temperature"),
        persona_id=effective_persona_id,
        custom_personas=custom_personas,
    )


@router.put("/v1/org/bots/{bot_id}/agent-config")
async def v1_org_update_agent_config(
    bot_id: str,
    payload: AgentConfigPayload,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    temperature = payload.temperature
    if temperature is not None and (temperature < 0 or temperature > 1):
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 1")
    config_dict = payload.model_dump(exclude_none=True)
    if not config_dict.get("persona_id"):
        config_dict["persona_id"] = get_default_persona_id()
    config_json = json.dumps(config_dict)
    bot_service().update_agent_config(bot_id, config_json)
    return {"status": "ok", "bot_id": bot_id}


# ── Personas ──────────────────────────────────────────────────────────────────

@router.get("/v1/personas", response_model=PersonaListResponse)
async def v1_list_personas(lang: str = "en"):
    """List all built-in personas with their categories. Pass ?lang=ja for Japanese."""
    return PersonaListResponse(
        personas=list_personas(lang=lang),
        categories=list_categories(lang=lang),
    )


def _default_prompt_fallback(business_name: str, lang: str = "en") -> str:
    """Fallback 3-part prompt when LLM or RAG is unavailable."""
    if lang == "ja":
        from domain.personas_ja import ABOUT_BUSINESS_JA, RESPONSE_RULES_JA
        return (
            f"## パーソナリティ\n"
            f"あなたは{business_name}のAIアシスタントで、親切でフレンドリーなガイドです。\n\n"
            f"## ビジネスについて\n"
            f"{business_name}はお客様に卓越したサービスを提供することに専念しています。"
            + RESPONSE_RULES_JA
        )
    from application.services.prompt_generation_service import _STANDARD_RESPONSE_RULES
    return (
        f"## Personality\n"
        f"You are {business_name}'s AI assistant, a helpful and friendly guide.\n\n"
        f"## About the Business\n"
        f"{business_name} is dedicated to providing excellent service to its customers."
        + _STANDARD_RESPONSE_RULES
    )


@router.post("/v1/org/bots/{bot_id}/generate-default-prompt")
async def v1_generate_default_prompt(
    bot_id: str,
    lang: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Query the bot's RAG corpus to build a curated, business-aware default system prompt.
    Returns the generated prompt text — caller saves it.
    Respects the bot's language setting for Japanese prompt generation.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")

    business_name = (bot.display_name or "").strip() or "the business"
    requested_lang = (lang or "").strip().lower()
    if requested_lang in ("ja", "jp"):
        bot_lang = "ja"
    elif requested_lang == "en":
        bot_lang = "en"
    else:
        bot_lang = _get_bot_language(bot)

    try:
        corpus = ensure_bot_corpus(bot.bot_id)
        import vertexai
        from infrastructure.clients.rag_client import retrieve_for_subquery, PROJECT_ID, RAG_LOCATION

        vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)

        # Pull overview content from the RAG corpus
        snippets: list[str] = []
        for subq in [
            f"What is {business_name}? What do they do?",
            f"What products or services does {business_name} offer?",
            f"About {business_name}",
        ]:
            try:
                results = retrieve_for_subquery(corpus, subq, top_k=3)
                for r in results:
                    text = (r.get("snippet") or "").strip()
                    if text and text not in snippets:
                        snippets.append(text)
                        if len(snippets) >= 6:
                            break
            except Exception:
                pass
            if len(snippets) >= 6:
                break

        # Use LLM to generate structured prompt from RAG snippets
        if snippets:
            from application.services.prompt_generation_service import generate_prompt_from_rag_content
            generated = generate_prompt_from_rag_content(snippets, business_name, lang=bot_lang)
            if generated:
                prompt_text = generated
            else:
                # LLM failed, use fallback template
                prompt_text = _default_prompt_fallback(business_name, lang=bot_lang)
        else:
            prompt_text = _default_prompt_fallback(business_name, lang=bot_lang)
    except Exception:
        prompt_text = _default_prompt_fallback(business_name, lang=bot_lang)

    return {"prompt": prompt_text, "business_name": business_name}


@router.post("/v1/org/bots/{bot_id}/generate-suggested-messages")
async def v1_generate_suggested_messages(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Auto-generate up to 3 suggested messages from the bot's trained content.
    Uses the RAG corpus to understand what the business offers, then asks
    an LLM to produce short, natural quick-reply labels.
    Saves directly to widget_config.suggestedMessages and returns them.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")

    business_name = (bot.display_name or "").strip() or "this business"
    bot_lang = _get_bot_language(bot)

    # Gather context from RAG corpus
    snippets: list[str] = []
    try:
        corpus = ensure_bot_corpus(bot.bot_id)
        import vertexai
        from infrastructure.clients.rag_client import retrieve_for_subquery, PROJECT_ID, RAG_LOCATION

        vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)

        queries = [
            f"What products or services does {business_name} offer?",
            f"What is {business_name}? What do they do?",
            f"Menu, pricing, or packages at {business_name}",
            f"How to book, order, or get started with {business_name}",
            f"Contact information, hours, or location of {business_name}",
        ]
        for subq in queries:
            try:
                results = retrieve_for_subquery(corpus, subq, top_k=3)
                for r in results:
                    text = (r.get("snippet") or "").strip()
                    if text and text not in snippets:
                        snippets.append(text)
                        if len(snippets) >= 10:
                            break
            except Exception:
                pass
            if len(snippets) >= 10:
                break
    except Exception:
        logger.exception("generate-suggested-messages: RAG retrieval failed for bot_id=%s", bot_id)

    # Also pull extracted topics if available
    topic_labels: list[str] = []
    try:
        from infrastructure.db.repositories import PostgresExtractedTopicRepository
        topic_repo = PostgresExtractedTopicRepository()
        topics = topic_repo.get_extracted_topics(org_id=resolved_org, bot_id=bot_id, active_only=True, limit=20)
        topic_labels = [t.topic for t in topics if t.topic]
    except Exception:
        pass

    if not snippets and not topic_labels:
        return {"suggestedMessages": [], "message": "No trained content found. Add sources first."}

    # Build LLM prompt
    context_parts = []
    if snippets:
        context_parts.append("=== Business Content ===\n" + "\n\n".join(snippets[:8]))
    if topic_labels:
        context_parts.append("=== Topics Covered ===\n" + ", ".join(topic_labels[:15]))

    context_block = "\n\n".join(context_parts)

    if bot_lang == "ja":
        generation_prompt = f"""「{business_name}」というビジネスを分析しています。
以下のコンテンツに基づいて、初めての訪問者が尋ねそうな3つのクイック返信メッセージを生成してください。

ルール：
- 重要：各メッセージは10文字以内（スペースと句読点を含む）にしてください
- 実際の顧客がタップするような短く自然な日本語のフレーズにしてください
- このビジネスの最も重要/人気のあるトピックをカバーしてください
- 一般的なものではなく、このビジネスが具体的に提供するものに合わせてください
- 引用符や番号は使わないでください
- 3つの文字列のJSON配列のみを返してください

良い例（すべて10文字以内）：
- レストラン: ["メニューを見る", "予約する", "本日のおすすめ"]
- ホテル: ["空室確認", "設備について", "予約する"]
- 美容院: ["料金プラン", "予約する", "営業時間"]
- ソフトウェア: ["機能一覧", "料金プラン", "デモを見る"]

{context_block}

正確に3つの文字列のJSON配列のみを返してください（各10文字以内）："""
    else:
        generation_prompt = f"""You are analysing a business called "{business_name}".
Based on the content below, generate exactly 3 suggested quick-reply messages that a first-time visitor would likely want to ask.

Rules:
- CRITICAL: Each message must be MAX 20 characters including spaces and punctuation
- Write them as short natural phrases a real customer would tap
- They should cover the most important/popular topics for THIS specific business
- Think about what a customer would ACTUALLY tap: services, pricing, hours, menu, booking, etc.
- Do NOT be generic. Tailor to what this business specifically offers
- Do NOT use quotes or numbering
- Return ONLY a JSON array of 3 strings, nothing else

Examples of good suggested messages (all under 20 chars):
- Restaurant: ["View the menu", "Make a reservation", "Today's specials"]
- Hotel: ["Room availability", "Amenities offered", "Book a room"]
- Salon: ["Services & prices", "Book appointment", "Opening hours"]
- Software: ["See features", "Pricing plans", "Get a demo"]

{context_block}

Return ONLY a valid JSON array of exactly 3 strings (each MUST be under 20 characters):"""

    # Call Gemini to generate
    try:
        from google import genai
        from google.genai import types as genai_types
        from infrastructure.clients.rag_client import PROJECT_ID, GENAI_LOCATION

        client = genai.Client(project=PROJECT_ID, location=GENAI_LOCATION)
        cfg = genai_types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=256,
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=generation_prompt)])],
            config=cfg,
        )
        raw = (resp.text or "").strip()

        # Parse JSON array from response
        # Strip markdown code fences if present
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        labels = json.loads(cleaned)
        if not isinstance(labels, list):
            raise ValueError("Expected JSON array")
        labels = [str(l).strip() for l in labels if str(l).strip()][:3]
    except Exception:
        logger.exception("generate-suggested-messages: LLM generation failed for bot_id=%s", bot_id)
        return {"suggestedMessages": [], "message": "Failed to generate messages. Try again."}

    if not labels:
        return {"suggestedMessages": [], "message": "Could not generate messages from content."}

    # Build suggestedMessages config
    import uuid
    suggested_messages = []
    for label in labels:
        suggested_messages.append({
            "id": str(uuid.uuid4())[:8],
            "label": label,
            "type": "ai_response",
        })

    # Save to widget_config
    existing_config: dict = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            existing_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    # Preserve existing escalate and show_menu messages
    prev_messages = existing_config.get("suggestedMessages") or []
    preserve_types = {"escalate", "show_menu"}
    preserved = [m for m in prev_messages if isinstance(m, dict) and m.get("type") in preserve_types]
    existing_config["suggestedMessages"] = suggested_messages + preserved
    bot_service().update_widget_config(bot_id, json.dumps(existing_config))

    return {"suggestedMessages": existing_config.get("suggestedMessages") or []}


@router.get("/v1/org/bots/{bot_id}/escalation-config", response_model=EscalationConfigResponse)
async def v1_org_get_escalation_config(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    cfg = _parse_escalation_config(getattr(bot, "escalation_config", None))
    return EscalationConfigResponse(
        enabled=cfg["enabled"],
        notify_enabled=cfg["notify_enabled"],
        notify_website=cfg["notify_website"],
        notify_instagram=cfg["notify_instagram"],
        notify_line=cfg["notify_line"],
        notification_emails=cfg["notification_emails"],
    )


@router.put("/v1/org/bots/{bot_id}/escalation-config", response_model=EscalationConfigResponse)
async def v1_org_update_escalation_config(
    bot_id: str,
    payload: EscalationConfigPayload,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    cfg = _parse_escalation_config(getattr(bot, "escalation_config", None))
    if payload.enabled is not None:
        cfg["enabled"] = bool(payload.enabled)
    if payload.notify_enabled is not None:
        cfg["notify_enabled"] = bool(payload.notify_enabled)
    if payload.notify_website is not None:
        cfg["notify_website"] = bool(payload.notify_website)
    if payload.notify_instagram is not None:
        cfg["notify_instagram"] = bool(payload.notify_instagram)
    if payload.notify_line is not None:
        cfg["notify_line"] = bool(payload.notify_line)
    if payload.notification_emails is not None:
        cfg["notification_emails"] = str(payload.notification_emails)
    bot_service().update_escalation_config(bot_id, json.dumps(cfg))
    return EscalationConfigResponse(
        enabled=cfg["enabled"],
        notify_enabled=cfg["notify_enabled"],
        notify_website=cfg["notify_website"],
        notify_instagram=cfg["notify_instagram"],
        notify_line=cfg["notify_line"],
        notification_emails=cfg["notification_emails"],
    )


@router.post("/v1/org/bots/{bot_id}/test-chat", response_model=TestChatResponse)
async def v1_org_test_chat(
    bot_id: str,
    payload: TestChatRequest,
    request: Request,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    corpus = ensure_bot_corpus(bot.bot_id)
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    session = conversation_service().get_or_create_session(
        bot_id=bot.bot_id,
        org_id=bot.org_id,
        channel="test",
        session_id=getattr(payload, "session_id", None),
        site_url=None,
        site_title=None,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="user",
        content=msg,
    )
    recent = conversation_service().list_recent_messages(session.session_id, limit=CONVERSATION_HISTORY_MESSAGES)
    conversation_context = _format_conversation_context(recent)
    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    widget_config_test: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config_test = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
    system_instruction = _resolve_system_instruction(
        agent_config,
        lang=_get_bot_language(bot),
        bot_name=getattr(bot, "display_name", "") or "",
        widget_config=widget_config_test,
    )
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None

    # Keep dashboard test-chat behavior aligned with live chat routes.
    extra_evidence_test: List[Dict[str, str]] = []
    availability_summary_test, _ = maybe_run_chat_availability(bot.bot_id, msg, widget_config_test)
    if availability_summary_test and _is_real_availability_summary(availability_summary_test):
        extra_evidence_test.append({"url": "Live availability check", "snippet": availability_summary_test})

    booking_url_test = _get_booking_url_for_chat(widget_config_test)
    if booking_url_test:
        extra_evidence_test.append({"url": booking_url_test, "snippet": f"To book or check availability, visit: {booking_url_test}"})

    url_bank_test = _get_url_bank_for_chat(widget_config_test)
    if url_bank_test:
        for it in url_bank_test:
            label = it["label"]
            url = it["url"]
            extra_evidence_test.append(
                {
                    "url": url,
                    "snippet": f"If the customer asks about {label}, share this link: [{label}]({url})",
                }
            )
        bank_lines = "\n".join([f"- {it['label']}: [{it['label']}]({it['url']})" for it in url_bank_test])
        bank_instruction_test = (
            "Answer links (use only when they match the customer’s question):\n"
            f"{bank_lines}\n\n"
            "If the question matches one of these topics, answer from that URL's content when it appears in the evidence, and include the matching link in your response.\n"
            "Write links as markdown like [Pricing](https://...) inside a normal sentence."
        )
        system_instruction = (
            f"{system_instruction}\n\n{bank_instruction_test}" if system_instruction else bank_instruction_test
        )

    reservation_cfg_test = get_reservation_config_from_widget(
        widget_config_test, lang=_get_bot_language(bot)
    )
    if reservation_cfg_test:
        extra_evidence_test.append({
            "url": reservation_cfg_test["url"],
            "snippet": f"Official online reservation page: {reservation_cfg_test['url']}",
        })
        system_instruction = (
            f"{system_instruction}\n\n{reservation_cfg_test['instruction']}"
            if system_instruction
            else reservation_cfg_test["instruction"]
        )

    asset_instruction_test = build_asset_instruction(bot.bot_id)
    if asset_instruction_test:
        system_instruction = (
            f"{system_instruction}\n\n{asset_instruction_test}" if system_instruction else asset_instruction_test
        )
    platform_asset_instruction_test = get_platform_asset_instructions(widget_config_test, lang=_get_bot_language(bot))
    if platform_asset_instruction_test:
        system_instruction = (
            f"{system_instruction}\n\n{platform_asset_instruction_test}" if system_instruction else platform_asset_instruction_test
        )

    result = run_vertex_rag(
        msg,
        rag_corpus=corpus,
        allowed_host=None,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        conversation_context=conversation_context or None,
        extra_evidence=extra_evidence_test if extra_evidence_test else None,
        bot_display_name=getattr(bot, "display_name", None),
    )
    sources = result.get("sources") or []
    citations = [Citation(url=str(s.get("url") or ""), snippet=str(s.get("excerpt") or "")) for s in sources]
    answer = str(result.get("answer") or "")
    if reservation_cfg_test:
        answer = ensure_canonical_reservation_url_in_text(
            answer, reservation_cfg_test["url"], reservation_cfg_test["domain_key"]
        )
    platform_features_test = get_platform_features_from_widget(widget_config_test)
    menu_extraction_enabled_test = platform_features_test.get("menu_extraction_enabled") if platform_features_test else False
    allowed_asset_types_test = {"menu_item"} if menu_extraction_enabled_test else None
    answer, marker_cards = resolve_asset_markers(
        answer, bot.bot_id, session.session_id, allowed_asset_types=allowed_asset_types_test
    )
    if marker_cards:
        asset_cards = marker_cards
    else:
        answer, asset_cards = process_answer_assets(
            answer,
            bot.bot_id,
            user_query=msg,
            session_id=session.session_id,
            allowed_asset_types=allowed_asset_types_test,
        )
    assets = [AssetCard(**c) for c in asset_cards]
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
        citations=[c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
    )
    return TestChatResponse(answer=answer, citations=citations, assets=assets, session_id=session.session_id)


@router.post("/v1/org/bots/{bot_id}/analytics/recompute", response_model=RecomputeResponse)
async def v1_org_recompute_analytics(
    bot_id: str,
    range: str = "30d",
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    data = analytics_service().recompute(org_id=resolved_org, bot_id=bot_id, range_str=range, from_day=from_day, to_day=to_day)
    return RecomputeResponse(ok=bool(data.get("ok")), start_day=str(data.get("start_day") or ""), end_day=str(data.get("end_day") or ""))


@router.get("/v1/org/bots/{bot_id}/analytics/summary", response_model=AnalyticsSummaryResponse)
async def v1_org_analytics_summary(
    bot_id: str,
    range: str = "30d",
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    s = analytics_service().summary(org_id=resolved_org, bot_id=bot_id, range_str=range, from_day=from_day, to_day=to_day)
    return AnalyticsSummaryResponse(**s.__dict__)


@router.get("/v1/org/bots/{bot_id}/analytics/timeseries", response_model=AnalyticsTimeseriesResponse)
async def v1_org_analytics_timeseries(
    bot_id: str,
    range: str = "30d",
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    data = analytics_service().timeseries(org_id=resolved_org, bot_id=bot_id, range_str=range, from_day=from_day, to_day=to_day)
    return AnalyticsTimeseriesResponse(**data)


@router.get("/v1/org/bots/{bot_id}/analytics/top-sources", response_model=TopSourcesResponse)
async def v1_org_analytics_top_sources(
    bot_id: str,
    range: str = "30d",
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    limit: int = 10,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    data = analytics_service().top_sources(org_id=resolved_org, bot_id=bot_id, range_str=range, limit=limit, from_day=from_day, to_day=to_day)
    return TopSourcesResponse(**data)


@router.get("/v1/org/bots/{bot_id}/analytics/topics", response_model=TopicsResponse)
async def v1_org_analytics_topics(
    bot_id: str,
    range: str = "30d",
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    limit: int = 20,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    data = analytics_service().topics(org_id=resolved_org, bot_id=bot_id, range_str=range, limit=limit, from_day=from_day, to_day=to_day)
    return TopicsResponse(**data)


@router.get("/v1/org/bots/{bot_id}/conversations", response_model=ConversationListResponse)
async def v1_org_list_conversations(
    bot_id: str,
    limit: int = 50,
    cursor: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    sessions = conversation_service().list_sessions(bot_id, limit=limit, before=cursor)
    total_count = conversation_service().count_sessions(bot_id)
    next_cursor = (
        f"{sessions[-1].last_active_at}|{sessions[-1].session_id}"
        if sessions and len(sessions) >= min(max(int(limit or 50), 1), 200)
        else None
    )
    return ConversationListResponse(
        bot_id=bot_id,
        sessions=[
            ConversationSessionResponse(
                session_id=s.session_id,
                bot_id=s.bot_id,
                channel=s.channel,
                status=s.status,
                title=s.title,
                site_url=s.site_url,
                site_title=s.site_title,
                message_count=s.message_count,
                started_at=s.started_at,
                last_active_at=s.last_active_at,
                ended_at=s.ended_at,
            )
            for s in sessions
        ],
        next_cursor=next_cursor,
        total_count=total_count,
    )


@router.get("/v1/org/bots/{bot_id}/conversations/search", response_model=ConversationSearchResponse)
async def v1_org_search_conversations(
    bot_id: str,
    q: Optional[str] = None,
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    status: Optional[str] = None,
    channel: Optional[str] = None,
    has_escalation: Optional[bool] = None,
    site_url: Optional[str] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    lim = max(1, min(int(limit or 50), 200))

    before_ts = None
    before_id = None
    if cursor:
        if "|" in cursor:
            before_ts, before_id = cursor.split("|", 1)
        else:
            before_ts = cursor

    def _day_to_iso(d: Optional[str], *, end: bool = False) -> Optional[str]:
        if not d:
            return None
        s = d.strip()
        if not s:
            return None
        # Expect YYYY-MM-DD; treat as UTC midnight bounds.
        try:
            parts = s.split("-")
            yy, mm, dd = int(parts[0]), int(parts[1]), int(parts[2])
            dt = datetime(yy, mm, dd, tzinfo=timezone.utc)
            if end:
                dt = dt + timedelta(days=1)
            return dt.isoformat()
        except Exception:
            return None

    start_iso = _day_to_iso(from_day, end=False)
    end_iso = _day_to_iso(to_day, end=True)

    qn = (q or "").strip()
    if len(qn) > 200:
        qn = qn[:200]

    con = get_connection()
    try:
        where = ["s.org_id=%s", "s.bot_id=%s"]
        params: list = [resolved_org, bot_id]

        if status:
            where.append("s.status=%s")
            params.append(status)
        if channel:
            where.append("s.channel=%s")
            params.append(channel)
        if site_url:
            where.append("s.site_url ILIKE %s")
            params.append(f"%{site_url.strip()}%")
        if start_iso:
            where.append("s.last_active_at >= %s")
            params.append(start_iso)
        if end_iso:
            where.append("s.last_active_at < %s")
            params.append(end_iso)
        if before_ts:
            where.append("(s.last_active_at, s.session_id) < (%s, %s)")
            params.append(before_ts)
            params.append(before_id or "")
        if has_escalation is True:
            where.append("EXISTS (SELECT 1 FROM conversation_escalations e WHERE e.session_id = s.session_id)")
        if has_escalation is False:
            where.append("NOT EXISTS (SELECT 1 FROM conversation_escalations e WHERE e.session_id = s.session_id)")
        if qn:
            where.append(
                "EXISTS (SELECT 1 FROM conversation_messages m WHERE m.session_id = s.session_id AND m.content ILIKE %s)"
            )
            params.append(f"%{qn}%")

        where_sql = " AND ".join(where)
        rows = con.execute(
            f"""
            SELECT s.session_id, s.bot_id, s.org_id, s.channel, s.status, s.title, s.site_url, s.site_title,
                   s.message_count, s.started_at, s.last_active_at, s.ended_at,
                   (
                     SELECT m.content
                     FROM conversation_messages m
                     WHERE m.session_id = s.session_id {("AND m.content ILIKE %s" if qn else "")}
                     ORDER BY m.created_at DESC
                     LIMIT 1
                   ) AS snippet
            FROM conversation_sessions s
            WHERE {where_sql}
            ORDER BY s.last_active_at DESC, s.session_id DESC
            LIMIT %s
            """,
            tuple(params + ([f"%{qn}%"] if qn else []) + [lim]),
        ).fetchall()

        sessions = []
        for r in rows or []:
            sessions.append(
                ConversationSearchSessionResponse(
                    session_id=r[0],
                    bot_id=r[1],
                    channel=r[3],
                    status=r[4],
                    title=r[5],
                    site_url=r[6],
                    site_title=r[7],
                    message_count=int(r[8] or 0),
                    started_at=r[9],
                    last_active_at=r[10],
                    ended_at=r[11],
                    snippet=(r[12] or "").strip() or None,
                )
            )

        # total_count is optional; compute only when cheap-ish (no keyword) to avoid heavy scans at scale.
        total_count = None
        if not qn:
            row = con.execute(f"SELECT COUNT(1) FROM conversation_sessions s WHERE {where_sql}", tuple(params)).fetchone()
            total_count = int(row[0]) if row else 0

        next_cursor = (
            f"{sessions[-1].last_active_at}|{sessions[-1].session_id}"
            if sessions and len(sessions) >= min(max(int(limit or 50), 1), 200)
            else None
        )
        return ConversationSearchResponse(bot_id=bot_id, sessions=sessions, next_cursor=next_cursor, total_count=total_count)
    finally:
        con.close()


@router.get("/v1/org/bots/{bot_id}/conversations/export.csv")
async def v1_org_export_conversations_csv(
    bot_id: str,
    q: Optional[str] = None,
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
    status: Optional[str] = None,
    channel: Optional[str] = None,
    has_escalation: Optional[bool] = None,
    site_url: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    # Reuse the search logic but stream results to CSV.
    qn = (q or "").strip()
    if len(qn) > 200:
        qn = qn[:200]

    def _day_to_iso(d: Optional[str], *, end: bool = False) -> Optional[str]:
        if not d:
            return None
        s = d.strip()
        if not s:
            return None
        try:
            parts = s.split("-")
            yy, mm, dd = int(parts[0]), int(parts[1]), int(parts[2])
            dt = datetime(yy, mm, dd, tzinfo=timezone.utc)
            if end:
                dt = dt + timedelta(days=1)
            return dt.isoformat()
        except Exception:
            return None

    start_iso = _day_to_iso(from_day, end=False)
    end_iso = _day_to_iso(to_day, end=True)

    where = ["s.org_id=%s", "s.bot_id=%s"]
    params: list = [resolved_org, bot_id]
    if status:
        where.append("s.status=%s")
        params.append(status)
    if channel:
        where.append("s.channel=%s")
        params.append(channel)
    if site_url:
        where.append("s.site_url ILIKE %s")
        params.append(f"%{site_url.strip()}%")
    if start_iso:
        where.append("s.last_active_at >= %s")
        params.append(start_iso)
    if end_iso:
        where.append("s.last_active_at < %s")
        params.append(end_iso)
    if has_escalation is True:
        where.append("EXISTS (SELECT 1 FROM conversation_escalations e WHERE e.session_id = s.session_id)")
    if has_escalation is False:
        where.append("NOT EXISTS (SELECT 1 FROM conversation_escalations e WHERE e.session_id = s.session_id)")
    if qn:
        where.append("EXISTS (SELECT 1 FROM conversation_messages m WHERE m.session_id = s.session_id AND m.content ILIKE %s)")
        params.append(f"%{qn}%")
    where_sql = " AND ".join(where)

    def _iter_csv():
        import csv
        import io

        con = get_connection()
        try:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                ["session_id", "channel", "status", "title", "site_url", "site_title", "message_count", "started_at", "last_active_at", "ended_at"]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

            rows = con.execute(
                f"""
                SELECT s.session_id, s.channel, s.status, COALESCE(s.title,''), COALESCE(s.site_url,''), COALESCE(s.site_title,''),
                       s.message_count, s.started_at, s.last_active_at, COALESCE(s.ended_at,'')
                FROM conversation_sessions s
                WHERE {where_sql}
                ORDER BY s.last_active_at DESC, s.session_id DESC
                """,
                tuple(params),
            ).fetchall()
            for r in rows or []:
                writer.writerow(list(r))
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)
        finally:
            con.close()

    return StreamingResponse(_iter_csv(), media_type="text/csv")


@router.get("/v1/org/bots/{bot_id}/escalations/counts", response_model=EscalationCountsResponse)
async def v1_org_escalation_counts(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    total = conversation_service().count_escalations(bot_id)
    open_count = conversation_service().count_open_escalations(bot_id)
    return EscalationCountsResponse(bot_id=bot_id, total=total, open=open_count)


@router.get("/v1/org/bots/{bot_id}/escalations", response_model=EscalationListResponse)
async def v1_org_list_escalations(
    bot_id: str,
    limit: int = 10,
    cursor: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    escalations = conversation_service().list_escalations(bot_id, limit=limit, before=cursor)
    total_count = conversation_service().count_escalations(bot_id)
    next_cursor = (
        f"{escalations[-1].created_at}|{escalations[-1].escalation_id}"
        if escalations and len(escalations) >= min(max(int(limit or 10), 1), 200)
        else None
    )
    return EscalationListResponse(
        bot_id=bot_id,
        escalations=[
            EscalationRecordResponse(
                escalation_id=e.escalation_id,
                bot_id=e.bot_id,
                session_id=e.session_id,
                visitor_email=e.visitor_email,
                status=e.status,
                created_at=e.created_at,
                details=e.details,
                title=e.session_title,
                site_url=e.site_url,
                site_title=e.site_title,
                last_active_at=e.last_active_at,
                session_status=e.session_status,
            )
            for e in escalations
        ],
        next_cursor=next_cursor,
        total_count=total_count,
    )


@router.get("/v1/org/bots/{bot_id}/escalations/{session_id}", response_model=EscalationRecordResponse)
async def v1_org_get_escalation_for_session(
    bot_id: str,
    session_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    record = conversation_service().get_escalation_for_session(bot_id, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="No escalation for this session")
    return EscalationRecordResponse(
        escalation_id=record.escalation_id,
        bot_id=record.bot_id,
        session_id=record.session_id,
        visitor_email=record.visitor_email,
        status=record.status,
        created_at=record.created_at,
        details=record.details,
        title=record.session_title,
        site_url=record.site_url,
        site_title=record.site_title,
        last_active_at=record.last_active_at,
        session_status=record.session_status,
    )


@router.post("/v1/org/bots/{bot_id}/escalations/{escalation_id}/status")
async def v1_org_update_escalation_status(
    bot_id: str,
    escalation_id: str,
    payload: EscalationStatusUpdateRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    status = (payload.status or "").strip().lower()
    if status not in {"open", "resolved"}:
        raise HTTPException(status_code=400, detail="Invalid escalation status")
    updated = conversation_service().update_escalation_status(bot_id, escalation_id, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Escalation not found")

    # When resolving, de-escalate any LINE / Instagram user session and notify user
    if status == "resolved":
        try:
            from infrastructure.db.repositories import PostgresLineUserSessionRepository, PostgresLineChannelRepository
            from infrastructure.clients.line_client import push_message as line_push

            esc_record = conversation_service().get_escalation_for_session(bot_id, "")
            # We need the session_id from the escalation -- look it up by escalation_id
            # The escalation record holds session_id, get it from the list
            all_escs = conversation_service().list_escalations(bot_id, limit=200)
            session_id = None
            for e in all_escs:
                if e.escalation_id == escalation_id:
                    session_id = e.session_id
                    break
            if session_id:
                line_session_repo = PostgresLineUserSessionRepository()
                mapping = line_session_repo.de_escalate_by_session_id(session_id)
                if mapping:
                    line_channel_repo = PostgresLineChannelRepository()
                    lc = line_channel_repo.get_by_bot_id(bot_id)
                    if lc and lc.is_active:
                        import asyncio
                        asyncio.ensure_future(
                            line_push(
                                mapping.line_user_id,
                                ["Your conversation has been resolved. You're now back with our AI assistant. How can I help you?"],
                                lc.line_channel_access_token,
                            )
                        )
        except Exception:
            pass  # Best-effort; don't block the status update

        # De-escalate Instagram session too
        try:
            from infrastructure.db.repositories import PostgresInstagramUserSessionRepository, PostgresInstagramChannelRepository
            from infrastructure.clients.instagram_client import send_message as ig_send

            if session_id:
                ig_session_repo = PostgresInstagramUserSessionRepository()
                ig_mapping = ig_session_repo.de_escalate_by_session_id(session_id)
                if ig_mapping:
                    ig_channel_repo = PostgresInstagramChannelRepository()
                    ig_ch = ig_channel_repo.get_by_bot_id(bot_id)
                    if ig_ch and ig_ch.is_active:
                        import asyncio
                        asyncio.ensure_future(
                            ig_send(
                                ig_mapping.ig_user_id,
                                "Your conversation has been resolved. You're now back with our AI assistant. How can I help you?",
                                ig_ch.page_access_token,
                            )
                        )
        except Exception:
            pass  # Best-effort; don't block the status update

    return {"status": status}


@router.get("/v1/org/bots/{bot_id}/conversations/{session_id}", response_model=ConversationDetailResponse)
async def v1_org_get_conversation(
    bot_id: str,
    session_id: str,
    limit: int = 200,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    session = conversation_service().get_session(session_id)
    if not session or session.bot_id != bot_id:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    messages = conversation_service().list_messages(session_id, limit=limit)
    return ConversationDetailResponse(
        bot_id=bot_id,
        session_id=session_id,
        messages=[
            ConversationMessageResponse(
                message_id=m.message_id,
                session_id=m.session_id,
                bot_id=m.bot_id,
                role=m.role,
                sender_name=m.sender_name,
                content=m.content,
                citations=[Citation(url=str(c.get("url") or ""), snippet=str(c.get("snippet") or "")) for c in (m.citations or [])],
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.post("/v1/org/bots/{bot_id}/conversations/{session_id}/end", response_model=ConversationEndResponse)
async def v1_org_end_conversation(
    bot_id: str,
    session_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    session = conversation_service().get_session(session_id)
    if not session or session.bot_id != bot_id:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    conversation_service().end_session(session_id, status="ended")
    return ConversationEndResponse(session_id=session_id, status="ended")


@router.get("/v1/pk/{publishable_key}/conversations", response_model=ConversationListResponse)
async def v1_pk_list_conversations(
    publishable_key: str,
    limit: int = 50,
    cursor: Optional[str] = None,
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    sessions = conversation_service().list_sessions(bot.bot_id, limit=limit, before=cursor)
    total_count = conversation_service().count_sessions(bot.bot_id)
    next_cursor = (
        f"{sessions[-1].last_active_at}|{sessions[-1].session_id}"
        if sessions and len(sessions) >= min(max(int(limit or 50), 1), 200)
        else None
    )
    return ConversationListResponse(
        bot_id=bot.bot_id,
        sessions=[
            ConversationSessionResponse(
                session_id=s.session_id,
                bot_id=s.bot_id,
                channel=s.channel,
                status=s.status,
                title=s.title,
                site_url=s.site_url,
                site_title=s.site_title,
                message_count=s.message_count,
                started_at=s.started_at,
                last_active_at=s.last_active_at,
                ended_at=s.ended_at,
            )
            for s in sessions
        ],
        next_cursor=next_cursor,
        total_count=total_count,
    )


@router.get("/v1/pk/{publishable_key}/conversations/{session_id}", response_model=ConversationDetailResponse)
async def v1_pk_get_conversation(
    publishable_key: str,
    session_id: str,
    limit: int = 200,
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    session = conversation_service().get_session(session_id)
    if not session or session.bot_id != bot.bot_id:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    messages = conversation_service().list_messages(session_id, limit=limit)
    return ConversationDetailResponse(
        bot_id=bot.bot_id,
        session_id=session_id,
        messages=[
            ConversationMessageResponse(
                message_id=m.message_id,
                session_id=m.session_id,
                bot_id=m.bot_id,
                role=m.role,
                sender_name=m.sender_name,
                content=m.content,
                citations=[Citation(url=str(c.get("url") or ""), snippet=str(c.get("snippet") or "")) for c in (m.citations or [])],
                created_at=m.created_at,
            )
            for m in messages
            if m.role != "system"
        ],
    )


@router.post("/v1/pk/{publishable_key}/conversations/{session_id}/end", response_model=ConversationEndResponse)
async def v1_pk_end_conversation(
    publishable_key: str,
    session_id: str,
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    session = conversation_service().get_session(session_id)
    if not session or session.bot_id != bot.bot_id:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    conversation_service().end_session(session_id, status="ended")
    return ConversationEndResponse(session_id=session_id, status="ended")


@router.post("/v1/pk/{publishable_key}/conversations/{session_id}/escalate", response_model=EscalationRecordResponse)
async def v1_pk_escalate_support(
    publishable_key: str,
    session_id: str,
    payload: EscalationCreateRequest,
    request: Request,
):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    session = conversation_service().get_session(session_id)
    if session_id == "new" or not session or session.bot_id != bot.bot_id:
        session = conversation_service().get_or_create_session(
            bot_id=bot.bot_id,
            org_id=bot.org_id,
            channel="chat",
            session_id=None,
            site_url=(payload.site_url or "").strip() or None,
            site_title=(payload.site_title or "").strip() or None,
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
        )
        conversation_service().add_message(
            session_id=session.session_id,
            bot_id=bot.bot_id,
            role="system",
            content="Escalated to support",
        )
    visitor_email = (payload.visitor_email or "").strip().lower()
    if not visitor_email or "@" not in visitor_email:
        raise HTTPException(status_code=400, detail="visitor_email is required")
    record = conversation_service().create_escalation(
        bot_id=bot.bot_id,
        session_id=session.session_id,
        visitor_email=visitor_email,
        details=(payload.details or "").strip() or None,
    )
    from infrastructure.email import maybe_send_escalation_email

    maybe_send_escalation_email(
        bot,
        session_id=session.session_id,
        channel="chat",
        visitor_email=visitor_email,
        details=(payload.details or "").strip() or None,
    )
    return EscalationRecordResponse(
        escalation_id=record.escalation_id,
        bot_id=record.bot_id,
        session_id=record.session_id,
        visitor_email=record.visitor_email,
        status=record.status,
        created_at=record.created_at,
        details=getattr(payload, "details", None),
    )


@router.delete("/v1/org/bots/{bot_id}")
async def v1_org_delete_bot(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    bot = bot_service().get_bot_record(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot_id")
    try:
        bot_service().delete_bot(bot_id)
        return {"status": "deleted", "bot_id": bot_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v1/org/bots/{bot_id}/domains", response_model=BotDomainListResponse)
async def v1_org_list_domains(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    domains = bot_service().list_domains(bot_id)
    return BotDomainListResponse(
        bot_id=bot_id,
        domains=[
            BotDomainRecordResponse(
                org_id=d.org_id,
                bot_id=d.bot_id,
                hostname=d.hostname,
                status=d.status,
                verification_token=d.verification_token,
                verified_at=d.verified_at,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in domains
        ],
    )


@router.post("/v1/org/bots/{bot_id}/domains", response_model=BotDomainAddResponse)
async def v1_org_add_domain(
    bot_id: str,
    payload: BotDomainAddRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    status, token = bot_service().add_domain(bot_id, payload.hostname)
    hostname = payload.hostname.strip().lower().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    return BotDomainAddResponse(
        bot_id=bot_id,
        hostname=hostname,
        status=status,
        verification_token=token,
        verification_url=_verification_url(hostname, token),
    )


@router.post("/v1/org/bots/{bot_id}/domains/{hostname}/verify", response_model=BotDomainVerifyResponse)
async def v1_org_verify_domain(
    bot_id: str,
    hostname: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    status, token = bot_service().add_domain(bot_id, hostname)
    ok, msg = _check_domain_verification(hostname, token)
    if ok:
        bot_service().mark_domain_verified(bot_id, hostname)
        return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status="verified", verified=True, message=msg)
    return BotDomainVerifyResponse(bot_id=bot_id, hostname=hostname, status=status, verified=False, message=msg)


@router.get("/v1/org/bots/{bot_id}/jobs", response_model=BotIndexJobListResponse)
async def v1_org_list_jobs(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    jobs = indexing_service().list_jobs_for_bot(bot_id)
    return BotIndexJobListResponse(
        bot_id=bot_id,
        jobs=[
            BotIndexJobResponse(
                job_id=j.job_id,
                url=j.url,
                hostname=j.hostname,
                stage=j.stage,
                pages_crawled=j.pages_crawled,
                docs_count=j.docs_count,
                gcs_prefix=j.gcs_prefix,
                last_error=j.last_error,
                created_at=j.created_at,
                updated_at=j.updated_at,
                crawled_urls=getattr(j, "crawled_urls", None) or [],
                source_id=getattr(j, "source_id", None),
            )
            for j in jobs
        ],
    )


@router.get("/v1/org/bots/{bot_id}/sources", response_model=BotSourceListResponse)
async def v1_org_list_sources(bot_id: str, org_id: Optional[str] = None, user=Depends(get_current_user)):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    sources = indexing_service().list_sources_for_bot(bot_id)
    return BotSourceListResponse(
        bot_id=bot_id,
        sources=[
            BotSourceResponse(
                source_id=s.source_id,
                bot_id=s.bot_id,
                type=s.type,
                config=s.config,
                display_name=s.display_name,
                created_at=s.created_at,
                updated_at=s.updated_at,
                sync_enabled=s.sync_enabled,
                sync_frequency=s.sync_frequency,
                sync_time_utc=s.sync_time_utc,
                sync_timezone=s.sync_timezone,
                last_synced_at=s.last_synced_at,
            )
            for s in sources
        ],
    )


@router.post("/v1/org/bots/{bot_id}/sources", response_model=BotSourceResponse)
async def v1_org_create_source(
    bot_id: str,
    payload: BotSourceCreateRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    source = indexing_service().create_source(
        bot_id, payload.type, payload.config or {}, payload.display_name
    )
    return BotSourceResponse(
        source_id=source.source_id,
        bot_id=source.bot_id,
        type=source.type,
        config=source.config,
        display_name=source.display_name,
        created_at=source.created_at,
        updated_at=source.updated_at,
        sync_enabled=source.sync_enabled,
        sync_frequency=source.sync_frequency,
        sync_time_utc=source.sync_time_utc,
        sync_timezone=source.sync_timezone,
        last_synced_at=source.last_synced_at,
    )


@router.post("/v1/org/bots/{bot_id}/sources/pdf", response_model=PdfSourceUploadResponse)
async def v1_org_upload_pdf_sources(
    bot_id: str,
    files: List[UploadFile] = File(...),
    display_name: Optional[str] = Form(default=None),
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Upload one or more PDFs as sources and start background ingestion immediately.
    Each file creates a BotSource(type='pdf') and a corresponding IndexJob.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    if not files:
        raise HTTPException(status_code=400, detail="Missing files")

    try:
        items: List[PdfSourceUploadItem] = []
        for f in files:
            filename = (getattr(f, "filename", None) or "").strip() or "document.pdf"
            content_type = (getattr(f, "content_type", None) or "").strip().lower()
            if content_type and content_type not in ("application/pdf", "application/x-pdf"):
                raise HTTPException(status_code=400, detail=f"Invalid content type for {filename}: {content_type}")
            data = await f.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"Empty file: {filename}")
            source, job_id = await indexing_service().create_pdf_source_and_start_ingest(
                bot_id=bot_id,
                filename=filename,
                pdf_bytes=data,
                display_name=display_name,
            )
            items.append(
                PdfSourceUploadItem(
                    source=BotSourceResponse(
                        source_id=source.source_id,
                        bot_id=source.bot_id,
                        type=source.type,
                        config=source.config,
                        display_name=source.display_name,
                        created_at=source.created_at,
                        updated_at=source.updated_at,
                    ),
                    job_id=job_id,
                    status="queued",
                )
            )
        return PdfSourceUploadResponse(bot_id=bot_id, items=items)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/org/bots/{bot_id}/sources/text", response_model=TextSourceUploadResponse)
async def v1_org_upload_text_sources(
    bot_id: str,
    body: TextSourceUploadRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Create one or more text/custom sources and start background ingestion.
    Each entry in body.entries creates a BotSource(type='text') and a corresponding IndexJob.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    if not body.entries:
        raise HTTPException(status_code=400, detail="Missing entries")

    try:
        items: List[TextSourceUploadItem] = []
        for entry in body.entries:
            content = (entry.content or "").strip()
            if not content:
                continue
            source, job_id = await indexing_service().create_text_source_and_start_ingest(
                bot_id=bot_id,
                content=content,
                title=entry.title,
            )
            items.append(TextSourceUploadItem(source_id=source.source_id, job_id=job_id, status="queued"))
        if not items:
            raise HTTPException(status_code=400, detail="No non-empty entries provided")
        return TextSourceUploadResponse(bot_id=bot_id, items=items)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


_ALLOWED_DOC_EXTENSIONS = {".txt", ".md", ".docx", ".doc"}
_ALLOWED_DOC_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
}


@router.post("/v1/org/bots/{bot_id}/sources/docs", response_model=DocsSourceUploadResponse)
async def v1_org_upload_docs_sources(
    bot_id: str,
    files: List[UploadFile] = File(...),
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Upload one or more text document files (.txt, .md, .docx, .doc) and start background ingestion.
    Each file creates a BotSource(type='docs') and a corresponding IndexJob.
    """
    import os as _os
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    if not files:
        raise HTTPException(status_code=400, detail="Missing files")

    try:
        items: List[DocsSourceUploadItem] = []
        for f in files:
            filename = (getattr(f, "filename", None) or "").strip() or "document.txt"
            ext = _os.path.splitext(filename.lower())[1]
            if ext not in _ALLOWED_DOC_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}. Allowed: .txt .md .docx .doc")
            content_type = (getattr(f, "content_type", None) or "application/octet-stream").strip().lower()
            data = await f.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"Empty file: {filename}")
            source, job_id = await indexing_service().create_docs_source_and_start_ingest(
                bot_id=bot_id,
                file_bytes=data,
                filename=filename,
                content_type=content_type,
            )
            items.append(DocsSourceUploadItem(source_id=source.source_id, job_id=job_id, status="queued"))
        return DocsSourceUploadResponse(bot_id=bot_id, items=items)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/org/bots/{bot_id}/sources/{source_id}", response_model=BotSourceResponse)
async def v1_org_get_source(
    bot_id: str,
    source_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    source = indexing_service().get_source(bot_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return BotSourceResponse(
        source_id=source.source_id,
        bot_id=source.bot_id,
        type=source.type,
        config=source.config,
        display_name=source.display_name,
        created_at=source.created_at,
        updated_at=source.updated_at,
        sync_enabled=source.sync_enabled,
        sync_frequency=source.sync_frequency,
        sync_time_utc=source.sync_time_utc,
        sync_timezone=source.sync_timezone,
        last_synced_at=source.last_synced_at,
    )


@router.put("/v1/org/bots/{bot_id}/sources/{source_id}/sync-settings", response_model=BotSourceResponse)
async def v1_org_update_source_sync_settings(
    bot_id: str,
    source_id: str,
    payload: BotSourceSyncSettingsRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    source = indexing_service().get_source(bot_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if (source.type or "").lower() != "url":
        raise HTTPException(status_code=400, detail="Sync is only supported for URL sources")
    from infrastructure.db.repositories import PostgresBotSourceRepository
    PostgresBotSourceRepository().update_sync_settings(
        bot_id, source_id, payload.sync_enabled, payload.sync_frequency, payload.sync_time_utc, payload.sync_timezone
    )
    updated = indexing_service().get_source(bot_id, source_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Source not found after update")
    return BotSourceResponse(
        source_id=updated.source_id,
        bot_id=updated.bot_id,
        type=updated.type,
        config=updated.config,
        display_name=updated.display_name,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
        sync_enabled=updated.sync_enabled,
        sync_frequency=updated.sync_frequency,
        sync_time_utc=updated.sync_time_utc,
        sync_timezone=updated.sync_timezone,
        last_synced_at=updated.last_synced_at,
    )


@router.post("/v1/org/bots/{bot_id}/sources/{source_id}/sync")
async def v1_org_sync_source(
    bot_id: str,
    source_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Manual sync: re-crawl a URL source to update RAG data."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    source = indexing_service().get_source(bot_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    if (source.type or "").lower() != "url":
        raise HTTPException(status_code=400, detail="Sync is only supported for URL sources")
    try:
        result = await indexing_service().start_indexing_for_source(bot_id, source_id)
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/org/bots/{bot_id}/sources/{source_id}")
async def v1_org_delete_source(
    bot_id: str,
    source_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    source = indexing_service().get_source(bot_id, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    indexing_service().delete_source(bot_id, source_id)
    return {"ok": True}


@router.post("/v1/org/url-discovery", response_model=UrlDiscoveryResponse)
async def v1_org_url_discovery(
    payload: UrlDiscoveryRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    _resolve_org_id(user, org_id)
    try:
        method = (payload.method or "auto").lower()
        urls = await url_discovery().discover(payload.url, method)
        if not urls and method == "sitemap":
            return UrlDiscoveryResponse(
                urls=[],
                error="No URLs found from sitemap. This could be because:\n• No sitemap.xml found in robots.txt\n• Sitemap is protected by CAPTCHA/bot detection\n• Sitemap is empty or invalid\n• Sitemap URLs are blocked\n\nTry using 'Automatic' discovery method instead (recommended).",
                method_used="sitemap"
            )
        return UrlDiscoveryResponse(urls=urls or [], method_used=method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log error but return error message instead of crashing
        import logging
        error_msg = str(e)
        logging.error(f"URL discovery error for {payload.url}: {error_msg}")
        method = (payload.method or "auto").lower()
        if method == "sitemap":
            return UrlDiscoveryResponse(
                urls=[],
                error=f"Sitemap discovery failed: {error_msg}\n\nTry using 'Automatic' discovery method instead (recommended).",
                method_used="sitemap"
            )
        return UrlDiscoveryResponse(
            urls=[],
            error=f"Automatic discovery failed: {error_msg}",
            method_used="auto"
        )


@router.post("/v1/org/url-discovery/stream")
async def v1_org_url_discovery_stream(
    payload: UrlDiscoveryRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Stream URL discovery as NDJSON so the dashboard can show progress.
    Uses Authorization header (so we can't use EventSource; we use fetch streaming).
    """
    _resolve_org_id(user, org_id)

    method = (payload.method or "auto").lower()

    async def _gen():
        try:
            async for evt in url_discovery().discover_stream(
                payload.url, method, max_depth=10, max_concurrent=10, max_urls=2000,
                max_duration_sec=payload.max_duration_sec,
            ):
                evt = dict(evt)
                if evt.get("type") in ("done", "start") and "method_used" not in evt:
                    evt["method_used"] = method
                yield json.dumps(evt, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {str(e)}", "method_used": method}) + "\n"
            yield json.dumps({"type": "done", "urls": [], "method_used": method}) + "\n"

    return StreamingResponse(
        _gen(),
        media_type="application/x-ndjson",
        headers={
            # Encourage proxies/servers not to buffer streaming responses.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/org/bots/{bot_id}/index")
async def v1_org_start_index(
    bot_id: str,
    payload: BotIndexRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        if payload.source_id and payload.source_id.strip():
            return await indexing_service().start_indexing_for_source(
                bot_id,
                payload.source_id.strip(),
                headless=payload.headless,
            )
        if not (payload.url and payload.url.strip()):
            raise HTTPException(status_code=400, detail="Provide url or source_id")
        return await indexing_service().start_indexing_for_bot(bot_id, payload.url.strip(), headless=payload.headless)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/org/bots/{bot_id}/sources/{source_id}/crawl-single")
async def v1_org_start_single_page_crawl(
    bot_id: str,
    source_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        return await indexing_service().start_single_page_crawl_for_source(bot_id, source_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/org/bots/{bot_id}/index/batch")
async def v1_org_start_index_batch(
    bot_id: str,
    payload: BotIndexBatchRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        return await indexing_service().start_indexing_batch_for_bot(bot_id, payload.urls)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _normalize_root_url(url: str) -> str:
    u = (url or "").strip()
    if u and not u.startswith(("http://", "https://")):
        return "https://" + u
    return u


@router.post("/v1/org/bots/{bot_id}/discovery-jobs")
async def v1_org_create_discovery_job(
    bot_id: str,
    payload: DiscoveryJobCreateRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Create a background discovery job and enqueue it. No time limit; results shown on Knowledge tab."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    root_url = _normalize_root_url(payload.url)
    if not root_url or not root_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid url")
    method = (payload.method or "auto").lower()
    if method not in ("auto", "sitemap"):
        method = "auto"
    job_id = "disc_" + uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    job = DiscoveryJob(
        job_id=job_id,
        bot_id=bot_id,
        root_url=root_url,
        method=method,
        status="queued",
        discovered_urls=[],
        error=None,
        celery_task_id=None,
        created_at=now,
        updated_at=now,
    )
    repo = PostgresDiscoveryJobRepository()
    repo.create(job)
    discovery_job_task.delay(job_id, bot_id, root_url, method)
    return {"job_id": job_id, "status": "queued"}


@router.get("/v1/org/bots/{bot_id}/discovery-jobs")
async def v1_org_list_discovery_jobs(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresDiscoveryJobRepository()
    jobs = repo.list_by_bot(bot_id)
    return DiscoveryJobListResponse(
        jobs=[
            DiscoveryJobResponse(
                job_id=j.job_id,
                bot_id=j.bot_id,
                root_url=j.root_url,
                method=j.method,
                status=j.status,
                discovered_urls=j.discovered_urls,
                discovered_count=len(j.discovered_urls),
                error=j.error,
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
            for j in jobs
        ]
    )


@router.get("/v1/org/bots/{bot_id}/discovery-jobs/{job_id}")
async def v1_org_get_discovery_job(
    bot_id: str,
    job_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresDiscoveryJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Discovery job not found")
    return DiscoveryJobResponse(
        job_id=job.job_id,
        bot_id=job.bot_id,
        root_url=job.root_url,
        method=job.method,
        status=job.status,
        discovered_urls=job.discovered_urls,
        discovered_count=len(job.discovered_urls),
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/v1/org/bots/{bot_id}/discovery-jobs/{job_id}/cancel")
async def v1_org_cancel_discovery_job(
    bot_id: str,
    job_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Cancel a running or queued background discovery job. Stops the process and hides discovery logs."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresDiscoveryJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Discovery job not found")
    if job.status not in ("queued", "running"):
        return {"status": "already_done", "job_id": job_id}
    if job.celery_task_id:
        try:
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception:
            pass
    job.status = "cancelled"
    repo.update(job)
    return {"status": "cancelled", "job_id": job_id}


@router.post("/v1/org/bots/{bot_id}/generate-prompt")
async def v1_org_generate_prompt(
    bot_id: str,
    lang: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Generate a system prompt from the bot's crawled homepage content.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)

    # 1. Find last crawl job with GCS prefix
    repo = PostgresIndexJobRepository()
    jobs = repo.list_jobs_for_bot(bot_id)
    valid_job = next((j for j in jobs if j.gcs_prefix), None)
    if not valid_job:
        raise HTTPException(status_code=400, detail="No crawled content found. Please train first.")

    # 2. Load homepage from GCS (reuse the same helper used by topic/asset extraction)
    from infrastructure.tasks.crawl_tasks import _load_docs_from_gcs_prefix
    all_documents = _load_docs_from_gcs_prefix(valid_job.gcs_prefix)
    if not all_documents:
        raise HTTPException(status_code=400, detail="No documents found in crawl data.")

    # Pick only the homepage document (matching root URL), not all crawled pages
    root_url = (valid_job.url or (valid_job.crawled_urls[0] if valid_job.crawled_urls else "")).strip().rstrip("/")
    homepage_doc = None
    for doc in all_documents:
        doc_url = (doc.get("url") or "").strip().rstrip("/")
        if doc_url and root_url and (doc_url == root_url or doc_url.rstrip("/") == root_url.rstrip("/")):
            homepage_doc = doc
            break
    if not homepage_doc:
        homepage_doc = all_documents[0]  # fallback to first doc

    homepage_content = homepage_doc.get("content", "")
    if not homepage_content:
        raise HTTPException(status_code=400, detail="Homepage content is empty.")

    # 3. Generate prompt
    try:
        # Fetch bot name to ensure accurate identity
        bot_repo = PostgresBotRepository()
        bot = bot_repo.get_bot(bot_id)
        business_name = bot.display_name if bot else ""
        requested_lang = (lang or "").strip().lower()
        if requested_lang in ("ja", "jp"):
            bot_lang = "ja"
        elif requested_lang == "en":
            bot_lang = "en"
        else:
            bot_lang = _get_bot_language(bot)

        url = valid_job.url or (valid_job.crawled_urls[0] if valid_job.crawled_urls else "")
        prompt = generate_prompt_from_content(
            homepage_content,
            root_url=url,
            business_name=business_name,
            lang=bot_lang,
        )
        if not prompt:
            raise HTTPException(status_code=500, detail="LLM failed to generate prompt.")
        return {"prompt": prompt}
    except HTTPException:
        raise
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=429, detail="AI quota exhausted. Please try again later.")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")




@router.get("/v1/org/bots/{bot_id}/index/status")
async def v1_org_index_status(
    bot_id: str,
    url: Optional[str] = None,
    job_id: Optional[str] = None,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        if job_id:
            return indexing_service().get_job_status(bot_id, job_id)
        if url:
            return indexing_service().get_job_status_by_hostname(bot_id, url)
        raise HTTPException(status_code=400, detail="Missing url or job_id parameter")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/org/bots/{bot_id}/index/cancel")
async def v1_org_cancel_index(
    bot_id: str,
    payload: BotIndexRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        return indexing_service().cancel_job(bot_id, payload.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========== Extracted Topics ==========

from api.schemas import (
    ExtractedTopicsResponse,
    ExtractedTopicItem,
    ExtractTopicsRequest,
    ExtractTopicsResponse,
    CreateTopicRequest,
    UpdateTopicRequest,
    DeleteTopicResponse,
    TopicJobItem,
    TopicJobsResponse,
    TopicUsageItem,
    TopicUsageSummaryResponse,
    TopicQuestionItem,
    TopicQuestionsResponse,
    SyncUrlBankRequest,
    ComputeMappingsResponse,
    AvailabilityRequest,
    AvailabilityJobItem,
    AvailabilityJobsResponse,
)
from application.services.topic_extraction_service import topic_extraction_service
from application.services.topic_question_service import topic_question_service
from infrastructure.db.repositories import PostgresTopicJobRepository, PostgresAvailabilityJobRepository
from domain.entities import AvailabilityJob
from infrastructure.availability.url_pattern import build_url, get_default_pattern, infer_pattern
from infrastructure.tasks.availability_tasks import availability_job_task


@router.get("/v1/org/bots/{bot_id}/extracted-topics", response_model=ExtractedTopicsResponse)
async def v1_org_get_extracted_topics(
    bot_id: str,
    active_only: bool = False,
    limit: int = 100,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get extracted topics for a bot."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    
    topics = topic_extraction_service().get_topics(
        org_id=resolved_org,
        bot_id=bot_id,
        active_only=active_only,
        limit=limit,
    )
    
    return ExtractedTopicsResponse(
        bot_id=bot_id,
        topics=[ExtractedTopicItem(**t) for t in topics],
        total_count=len(topics),
    )


@router.post("/v1/org/bots/{bot_id}/extracted-topics", response_model=ExtractedTopicItem)
async def v1_org_create_topic(
    bot_id: str,
    payload: CreateTopicRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Create a single topic (e.g. from Manage Topics "Add topic" in a box)."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    created = topic_extraction_service().create_topic(
        org_id=resolved_org,
        bot_id=bot_id,
        topic=payload.topic,
        category=payload.category,
    )
    if not created:
        raise HTTPException(status_code=400, detail="Invalid topic (empty or duplicate)")
    return ExtractedTopicItem(**created)


@router.post("/v1/org/bots/{bot_id}/extracted-topics/extract", response_model=ExtractTopicsResponse)
async def v1_org_extract_topics(
    bot_id: str,
    payload: ExtractTopicsRequest = ExtractTopicsRequest(),
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """
    Trigger topic extraction from RAG content for a bot.
    This fetches content from GCS (crawled documents) and extracts topics using LLM.
    """
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    
    service = topic_extraction_service()
    
    # Get the latest completed index job to find GCS prefix
    from infrastructure.db.repositories import PostgresIndexJobRepository
    job_repo = PostgresIndexJobRepository()
    jobs = job_repo.list_jobs_for_bot(bot_id)
    
    # Find a completed job with GCS content
    gcs_prefix = None
    for job in jobs:
        if job.stage in ('done', 'import_submitted', 'prompt_queued', 'prompt_generating') and job.gcs_prefix:
            gcs_prefix = job.gcs_prefix
            break
    
    if not gcs_prefix:
        # No crawled content, return empty
        return ExtractTopicsResponse(
            bot_id=bot_id,
            topics_extracted=0,
            topics=[],
        )
    
    # Fetch content from GCS
    try:
        from google.cloud import storage
        from common.config import config
        import os
        import traceback
        
        bucket_raw = config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")
        if not bucket_raw:
            print(f"[TopicExtraction] GCS_BUCKET not configured, cannot extract topics")
            return ExtractTopicsResponse(
                bot_id=bot_id,
                topics_extracted=0,
                topics=[],
            )
        
        # Handle bucket name with optional path prefix (e.g., "bucket-name/prefix/")
        # Note: gcs_prefix from DB already includes the full path (e.g., "saas/org-xxx/bots/...")
        # so we only need to extract the bucket name, not prepend any prefix
        bucket_parts = bucket_raw.strip("/").split("/", 1)
        bucket_name = bucket_parts[0]
        
        print(f"[TopicExtraction] Fetching content from GCS bucket={bucket_name}, prefix={gcs_prefix}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List and read markdown files from the GCS prefix (no max_results to find all .md in sub-dirs)
        documents = []
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        print(f"[TopicExtraction] Found {len(blobs)} blobs in GCS")
        
        for blob in blobs:
            if blob.name.endswith('.md'):
                try:
                    content = blob.download_as_text(encoding="utf-8")
                    # Extract URL from content (first line is usually "Source URL: ...")
                    url = ""
                    if content.startswith("Source URL:"):
                        first_line = content.split('\n')[0]
                        url = first_line.replace("Source URL:", "").strip()
                    documents.append({"url": url, "content": content})
                except Exception as e:
                    print(f"[TopicExtraction] Error reading blob {blob.name}: {e}")
                    continue
        
        print(f"[TopicExtraction] Read {len(documents)} documents from GCS")
        
        if not documents:
            return ExtractTopicsResponse(
                bot_id=bot_id,
                topics_extracted=0,
                topics=[],
            )
        
        # Extract topics from documents
        print(f"[TopicExtraction] Starting LLM extraction...")
        extracted = service.extract_topics_from_documents(
            org_id=resolved_org,
            bot_id=bot_id,
            documents=documents,
            clear_existing=payload.clear_existing,
        )
        print(f"[TopicExtraction] Extracted {len(extracted)} topics")
        
        return ExtractTopicsResponse(
            bot_id=bot_id,
            topics_extracted=len(extracted),
            topics=[ExtractedTopicItem(**t) for t in extracted],
        )
        
    except Exception as e:
        import traceback
        print(f"[TopicExtraction] Error extracting topics: {e}")
        traceback.print_exc()
        # Return existing topics on error
        topics = service.get_topics(org_id=resolved_org, bot_id=bot_id, limit=100)
        return ExtractTopicsResponse(
            bot_id=bot_id,
            topics_extracted=len(topics),
            topics=[ExtractedTopicItem(**t) for t in topics],
        )


@router.patch("/v1/org/bots/{bot_id}/extracted-topics/{topic_id}", response_model=ExtractedTopicItem)
async def v1_org_update_topic(
    bot_id: str,
    topic_id: str,
    payload: UpdateTopicRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Update a topic's active status or category."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    
    updated = topic_extraction_service().update_topic(
        topic_id=topic_id,
        is_active=payload.is_active,
        category=payload.category,
    )
    
    if not updated:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return ExtractedTopicItem(**updated)


@router.delete("/v1/org/bots/{bot_id}/extracted-topics/{topic_id}", response_model=DeleteTopicResponse)
async def v1_org_delete_topic(
    bot_id: str,
    topic_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Delete a topic."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    
    deleted = topic_extraction_service().delete_topic(topic_id=topic_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return DeleteTopicResponse(ok=True, topic_id=topic_id)


@router.get("/v1/org/bots/{bot_id}/topic-jobs", response_model=TopicJobsResponse)
async def v1_org_list_topic_jobs(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresTopicJobRepository()
    jobs = repo.list_by_bot(bot_id) or []
    return TopicJobsResponse(
        bot_id=bot_id,
        jobs=[TopicJobItem(**job.__dict__) for job in jobs],
    )


@router.get("/v1/org/bots/{bot_id}/topic-jobs/{job_id}", response_model=TopicJobItem)
async def v1_org_get_topic_job(
    bot_id: str,
    job_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresTopicJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Topic job not found")
    return TopicJobItem(**job.__dict__)


# ========== Topic Usage & Question Mapping ==========

@router.get("/v1/org/bots/{bot_id}/topics/usage-summary", response_model=TopicUsageSummaryResponse)
async def v1_org_topic_usage_summary(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Return topic usage counts for the donut chart (how many questions per topic)."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    summary = topic_question_service().get_topic_usage_summary(
        org_id=resolved_org,
        bot_id=bot_id,
    )
    return TopicUsageSummaryResponse(
        bot_id=bot_id,
        topics=[TopicUsageItem(**t) for t in summary["topics"]],
        total_questions=summary["total_questions"],
    )


@router.post("/v1/org/bots/{bot_id}/topics/compute-mappings", response_model=ComputeMappingsResponse)
async def v1_org_compute_topic_mappings(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Scan conversations and map questions to topics. Call on dashboard load or on refresh."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    new_count = topic_question_service().compute_topic_mappings(
        org_id=resolved_org,
        bot_id=bot_id,
    )
    return ComputeMappingsResponse(bot_id=bot_id, new_mappings=new_count)


@router.get("/v1/org/bots/{bot_id}/topics/{topic_id}/questions", response_model=TopicQuestionsResponse)
async def v1_org_topic_questions(
    bot_id: str,
    topic_id: str,
    limit: int = 50,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get questions linked to a specific topic."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    questions = topic_question_service().get_questions_for_topic(
        topic_id=topic_id,
        bot_id=bot_id,
        limit=limit,
    )
    return TopicQuestionsResponse(
        topic_id=topic_id,
        questions=[TopicQuestionItem(**q) for q in questions],
    )


@router.post("/v1/org/bots/{bot_id}/topics/sync-url-bank", response_model=ExtractedTopicsResponse)
async def v1_org_sync_url_bank_topics(
    bot_id: str,
    payload: SyncUrlBankRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Sync URL bank entries as topics. Call when URL bank is saved.
    If url_bank exists with URLs, triggers indexing so the agent can answer from that content when topics match."""
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    synced = topic_extraction_service().sync_url_bank_topics(
        org_id=resolved_org,
        bot_id=bot_id,
        url_bank=payload.url_bank,
    )

    # If url_bank exists with URLs, index them so the agent can answer from that content when topics match
    if payload.url_bank:
        urls_to_index = []
        for entry in payload.url_bank:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").strip()
            if not url:
                continue
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            urls_to_index.append(url)
        if urls_to_index:
            async def _index_url_bank():
                try:
                    await indexing_service().start_indexing_batch_for_bot(bot_id, urls_to_index)
                except (PermissionError, ValueError, RuntimeError) as e:
                    logger.warning("URL bank indexing skipped for bot_id=%s: %s", bot_id, e)

            asyncio.create_task(_index_url_bank())

    return ExtractedTopicsResponse(
        bot_id=bot_id,
        topics=[ExtractedTopicItem(**t) for t in synced],
        total_count=len(synced),
    )


@router.post("/v1/org/bots/{bot_id}/availability", response_model=AvailabilityJobItem)
async def v1_org_start_availability_job(
    bot_id: str,
    payload: AvailabilityRequest,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    if not payload.url:
        raise HTTPException(status_code=400, detail="Missing url")

    url_input = payload.url.strip()
    if not url_input.startswith(("http://", "https://")):
        url_input = "https://" + url_input

    check_in = payload.check_in or ""
    check_out = payload.check_out or ""
    adults = payload.adults or 2
    children = payload.children or 0
    rooms = payload.rooms or 1

    # Load widget_config for bookingUrlPattern
    bot = bot_service().get_bot_record(bot_id)
    widget_config: Dict[str, Any] = {}
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass

    booking_pattern = widget_config.get("bookingUrlPattern") if isinstance(widget_config, dict) else None
    resolved_url: str
    debug_info: Dict[str, Any] = {
        "url_input": url_input,
        "check_in": check_in,
        "check_out": check_out,
        "adults": adults,
        "children": children,
        "rooms": rooms,
    }

    if check_in and check_out:
        # User provided dates: build URL from pattern
        if booking_pattern and isinstance(booking_pattern, dict) and booking_pattern.get("param_mapping"):
            debug_info["pattern_source"] = "saved_bookingUrlPattern"
            debug_info["pattern"] = booking_pattern
            resolved_url = build_url(
                booking_pattern,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                rooms=rooms,
            )
        else:
            inferred = infer_pattern(url_input)
            debug_info["pattern_inferred"] = inferred
            if inferred:
                debug_info["pattern_source"] = "infer_pattern"
                bot_service().update_widget_config(
                    bot_id,
                    json.dumps({
                        **widget_config,
                        "bookingUrlPattern": inferred,
                        "bookingTestUrl": url_input,
                    }),
                )
                resolved_url = build_url(
                    inferred,
                    check_in=check_in,
                    check_out=check_out,
                    adults=adults,
                    children=children,
                    rooms=rooms,
                )
            else:
                default_pat = get_default_pattern(url_input)
                debug_info["pattern_source"] = "get_default_pattern"
                debug_info["pattern"] = default_pat
                resolved_url = build_url(
                    default_pat,
                    check_in=check_in,
                    check_out=check_out,
                    adults=adults,
                    children=children,
                    rooms=rooms,
                )
    else:
        # User pasted full URL: use as-is and infer/save pattern for future
        resolved_url = url_input
        debug_info["pattern_source"] = "url_as_is"
        inferred = infer_pattern(url_input)
        debug_info["pattern_inferred"] = inferred
        if inferred:
            bot_service().update_widget_config(
                bot_id,
                json.dumps({
                    **widget_config,
                    "bookingUrlPattern": inferred,
                    "bookingTestUrl": url_input,
                }),
            )

    now = datetime.now(timezone.utc).isoformat()
    job_id = "avail_" + secrets.token_urlsafe(16).replace("-", "_").replace(".", "_")
    screenshots_dir = f"/app/backend/data/availability/{job_id}"
    debug_info["resolved_url"] = resolved_url
    os.makedirs(screenshots_dir, exist_ok=True)
    try:
        with open(os.path.join(screenshots_dir, "debug_info.json"), "w", encoding="utf-8") as f:
            json.dump(debug_info, f, indent=2)
    except Exception:
        pass
    job = AvailabilityJob(
        job_id=job_id,
        org_id=resolved_org,
        bot_id=bot_id,
        url=resolved_url,
        status="queued",
        question=(payload.question or None),
        summary=None,
        last_error=None,
        max_seconds=max(15, min(payload.max_seconds or 60, 180)),
        steps_count=0,
        screenshots_dir=screenshots_dir,
        celery_task_id=None,
        created_at=now,
        updated_at=now,
    )
    repo = PostgresAvailabilityJobRepository()
    repo.create(job)
    task = availability_job_task.delay(
        job_id=job_id,
        bot_id=bot_id,
        org_id=resolved_org,
        url=resolved_url,
        question=payload.question or "",
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        children=children,
        rooms=rooms,
        max_seconds=job.max_seconds,
        screenshots_dir=screenshots_dir,
    )
    job.celery_task_id = task.id if task else None
    repo.update(job)
    return AvailabilityJobItem(**job.__dict__)


@router.get("/v1/org/bots/{bot_id}/availability", response_model=AvailabilityJobsResponse)
async def v1_org_list_availability_jobs(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresAvailabilityJobRepository()
    jobs = repo.list_by_bot(bot_id)
    return AvailabilityJobsResponse(
        bot_id=bot_id,
        jobs=[AvailabilityJobItem(**job.__dict__) for job in jobs],
    )


@router.get("/v1/org/bots/{bot_id}/availability/{job_id}", response_model=AvailabilityJobItem)
async def v1_org_get_availability_job(
    bot_id: str,
    job_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresAvailabilityJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Availability job not found")
    return AvailabilityJobItem(**job.__dict__)


@router.get("/v1/org/bots/{bot_id}/availability/{job_id}/raw")
async def v1_org_get_availability_raw(
    bot_id: str,
    job_id: str,
    format: str = "text",
    max_chars: int = 0,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresAvailabilityJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Availability job not found")
    fmt = (format or "text").lower()
    if fmt not in ("text", "html", "debug"):
        raise HTTPException(status_code=400, detail="format must be text, html, or debug")
    if fmt == "debug":
        path = os.path.join(job.screenshots_dir, "availability_debug.log") if job.screenshots_dir else None
    elif fmt == "text":
        path = job.raw_text_path
    else:
        path = job.raw_html_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found (format={fmt})")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read raw file: {exc}")
    if max_chars and max_chars > 0:
        content = content[: max_chars]
    return {"format": fmt, "content": content}


# ========== Booking Link Jobs ==========

@router.get("/v1/org/bots/{bot_id}/booking-links", response_model=BookingLinkJobsResponse)
async def v1_org_list_booking_link_jobs(
    bot_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresBookingLinkJobRepository()
    jobs = repo.list_by_bot(bot_id)
    return BookingLinkJobsResponse(
        bot_id=bot_id,
        jobs=[
            BookingLinkJobItem(
                job_id=job.job_id,
                bot_id=job.bot_id,
                index_job_id=job.index_job_id,
                root_url=job.root_url,
                status=job.status,
                links=job.links,
                error=job.error,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
            for job in jobs
        ],
    )


@router.get("/v1/org/bots/{bot_id}/booking-links/{job_id}", response_model=BookingLinkJobItem)
async def v1_org_get_booking_link_job(
    bot_id: str,
    job_id: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    repo = PostgresBookingLinkJobRepository()
    job = repo.get(bot_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Booking link job not found")
    return BookingLinkJobItem(
        job_id=job.job_id,
        bot_id=job.bot_id,
        index_job_id=job.index_job_id,
        root_url=job.root_url,
        status=job.status,
        links=job.links,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ========== Admin Endpoints ==========

@router.post("/v1/admin/reset/gcs")
async def v1_admin_reset_gcs(user=Depends(require_super_admin)):
    try:
        bucket_name, base_prefix, deleted = delete_gcs_objects(allow_root=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "bucket": bucket_name, "prefix": base_prefix, "deleted_objects": deleted}


@router.post("/v1/admin/reset/rag")
async def v1_admin_reset_rag(user=Depends(require_super_admin)):
    deleted = delete_rag_corpora()
    return {"status": "ok", "deleted_corpora": deleted}

