import time
import urllib.request
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.deps.auth import get_current_user, require_org_admin, require_super_admin
from api.schemas import (
    AgentConfigPayload,
    AgentConfigResponse,
    BotCreateRequest,
    BotCreateResponse,
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
    WidgetChatRequest,
    WidgetChatResponse,
    WidgetConfigUpdate,
)
from application.auth.jwt_auth import is_super_admin
from common.config import config
from application.services.conversation_service import CONVERSATION_HISTORY_MESSAGES
from common.di.container import bot_service, conversation_service, indexing_service, org_service, url_discovery, user_service
from common.di.container import analytics_service
from common.logging.chat_debug import chat_debug_emit
from infrastructure.clients.rag_client import run_vertex_rag, run_vertex_rag_stream
from infrastructure.services.indexing_service import ensure_bot_corpus
from infrastructure.services.reset_service import delete_gcs_objects, delete_rag_corpora
from infrastructure.db.repositories import PostgresDiscoveryJobRepository
from infrastructure.db.connection import get_connection
from infrastructure.tasks.discovery_tasks import discovery_job_task
from domain.entities import DiscoveryJob

router = APIRouter()

# Per-message truncation for conversation context (keeps prompt size bounded).
_CONVERSATION_CONTEXT_MAX_CHARS = 500


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
        return {"enabled": False, "notify_enabled": False, "notification_emails": ""}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {"enabled": False, "notify_enabled": False, "notification_emails": ""}
    return {
        "enabled": bool(data.get("enabled")),
        "notify_enabled": bool(data.get("notify_enabled")),
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
        return await indexing_service().start_indexing_for_bot(bot_id, payload.url)
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
        return await indexing_service().start_indexing_for_bot(bot.bot_id, payload.url)
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
    if not getattr(bot, "widget_config", None) or not (bot.widget_config or "").strip():
        return {}
    try:
        return json.loads(bot.widget_config)
    except (TypeError, ValueError):
        return {}


@router.get("/v1/pk/{publishable_key}/escalation-config", response_model=EscalationConfigResponse)
async def v1_pk_escalation_config(publishable_key: str):
    bot = bot_service().get_bot_by_publishable_key(publishable_key)
    if not bot:
        raise HTTPException(status_code=404, detail="Unknown bot publishable key")
    cfg = _parse_escalation_config(getattr(bot, "escalation_config", None))
    return EscalationConfigResponse(
        enabled=cfg["enabled"],
        notify_enabled=cfg["notify_enabled"],
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

    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = agent_config.get("instructions") if agent_config else None
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
    result = run_vertex_rag(
        query,
        rag_corpus=corpus,
        allowed_host=allowed_host or None,
        debug_cb=_rag_dbg,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        conversation_context=conversation_context or None,
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
        answer = f"I can’t find that in the indexed content for {host_label}. Try asking about something on the site, or re-run Crawl."
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
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
        citations=[c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
    )
    return WidgetChatResponse(answer=answer, citations=citations, session_id=session.session_id)


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

    agent_config = {}
    if getattr(bot, "agent_config", None) and (bot.agent_config or "").strip():
        try:
            agent_config = json.loads(bot.agent_config)
        except (TypeError, ValueError):
            pass
    system_instruction = agent_config.get("instructions") if agent_config else None
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

    async def _gen():
        yield json.dumps({"type": "meta", "session_id": session.session_id}, ensure_ascii=False) + "\n"
        try:
            for evt in run_vertex_rag_stream(
                query,
                rag_corpus=corpus,
                allowed_host=allowed_host or None,
                debug_cb=_rag_dbg,
                system_instruction=system_instruction,
                model_name=model_name,
                temperature=temperature,
                conversation_context=conversation_context or None,
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
                        chat_debug_emit(
                            {
                                "type": "chat_response",
                                "trace_id": trace_id,
                                "answer": str(evt.get("answer") or ""),
                                "citations": citations,
                            }
                        )
                        answer = str(evt.get("answer") or "")
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
    widget_config = None
    if getattr(bot, "widget_config", None) and (bot.widget_config or "").strip():
        try:
            widget_config = json.loads(bot.widget_config)
        except (TypeError, ValueError):
            pass
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
    config_dict = payload.model_dump(exclude_none=True)
    config_json = json.dumps(config_dict)
    bot_service().update_widget_config(bot_id, config_json)
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
    return AgentConfigResponse(
        model_id=agent_config.get("model_id"),
        instructions=agent_config.get("instructions"),
        temperature=agent_config.get("temperature"),
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
    config_json = json.dumps(config_dict)
    bot_service().update_agent_config(bot_id, config_json)
    return {"status": "ok", "bot_id": bot_id}


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
    if payload.notification_emails is not None:
        cfg["notification_emails"] = str(payload.notification_emails)
    bot_service().update_escalation_config(bot_id, json.dumps(cfg))
    return EscalationConfigResponse(
        enabled=cfg["enabled"],
        notify_enabled=cfg["notify_enabled"],
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
    system_instruction = agent_config.get("instructions") if agent_config else None
    model_name = agent_config.get("model_id") if agent_config else None
    temperature = agent_config.get("temperature") if agent_config else None
    result = run_vertex_rag(
        msg,
        rag_corpus=corpus,
        allowed_host=None,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        conversation_context=conversation_context or None,
    )
    sources = result.get("sources") or []
    citations = [Citation(url=str(s.get("url") or ""), snippet=str(s.get("excerpt") or "")) for s in sources]
    answer = str(result.get("answer") or "")
    conversation_service().add_message(
        session_id=session.session_id,
        bot_id=bot.bot_id,
        role="bot",
        content=answer,
        citations=[c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
    )
    return TestChatResponse(answer=answer, citations=citations, session_id=session.session_id)


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
    cfg = _parse_escalation_config(getattr(bot, "escalation_config", None))
    if not cfg.get("enabled"):
        raise HTTPException(status_code=400, detail="Escalations are disabled for this bot")
    record = conversation_service().create_escalation(
        bot_id=bot.bot_id,
        session_id=session.session_id,
        visitor_email=visitor_email,
        details=(payload.details or "").strip() or None,
    )
    # TODO: send email notification when notify_enabled is true.
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
    )


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
    )


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
            return await indexing_service().start_indexing_for_source(bot_id, payload.source_id.strip())
        if not (payload.url and payload.url.strip()):
            raise HTTPException(status_code=400, detail="Provide url or source_id")
        return await indexing_service().start_indexing_for_bot(bot_id, payload.url.strip())
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
)
from application.services.topic_extraction_service import topic_extraction_service


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
        if job.stage in ('done', 'import_submitted') and job.gcs_prefix:
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
        
        # List and read markdown files from the GCS prefix
        documents = []
        blobs = list(bucket.list_blobs(prefix=gcs_prefix, max_results=50))
        print(f"[TopicExtraction] Found {len(blobs)} blobs in GCS")
        
        for blob in blobs:
            if blob.name.endswith('.md'):
                try:
                    content = blob.download_as_text()
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
