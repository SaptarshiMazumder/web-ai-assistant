import time
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from api.deps.auth import get_current_user, require_org_admin, require_super_admin
from api.schemas import (
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
    BotIndexRequest,
    BotListResponse,
    BotSummary,
    Citation,
    OrgCreateRequest,
    OrgListResponse,
    OrgMemberAddRequest,
    OrgMemberResponse,
    OrgMembersListResponse,
    OrgSelfResponse,
    OrgSummary,
    OrgUpdateRequest,
    WidgetChatRequest,
    WidgetChatResponse,
)
from application.auth.jwt_auth import is_super_admin
from common.config import config
from common.di.container import bot_service, org_service, user_service
from common.logging.chat_debug import chat_debug_emit
from infrastructure.clients.rag_client import run_vertex_rag
from infrastructure.services.indexing_service import (
    cancel_index_for_bot,
    ensure_bot_corpus,
    get_index_status_for_bot,
    list_index_jobs_for_bot,
    start_index_for_bot,
)
from infrastructure.services.reset_service import delete_gcs_objects, delete_rag_corpora

router = APIRouter()


def _require_admin_key(x_admin_key: Optional[str]) -> None:
    required = (getattr(config, "ADMIN_API_KEY", None) or "").strip()
    if not required:
        return
    if (x_admin_key or "").strip() != required:
        raise HTTPException(status_code=401, detail="Missing/invalid admin key")


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
        return await start_index_for_bot(bot_id, payload.url)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/bots/{bot_id}/jobs", response_model=BotIndexJobListResponse)
async def v1_list_jobs(bot_id: str, x_admin_key: Optional[str] = Header(default=None)):
    _require_admin_key(x_admin_key)
    jobs = list_index_jobs_for_bot(bot_id)
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
        return get_index_status_for_bot(bot_id, url)
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
        return cancel_index_for_bot(bot_id, payload.url)
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
        return await start_index_for_bot(bot.bot_id, payload.url)
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
        return get_index_status_for_bot(bot.bot_id, url)
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
        return cancel_index_for_bot(bot.bot_id, payload.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
            allowed_host = (urlparse(site_url).hostname or "").lower().split(":")[0]
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

    result = run_vertex_rag(query, rag_corpus=corpus, allowed_host=allowed_host or None, debug_cb=_rag_dbg)
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
        return WidgetChatResponse(
            answer=f"I can’t find that in the indexed content for {host_label}. Try asking about something on the site, or re-run Crawl.",
            citations=[],
        )

    chat_debug_emit(
        {
            "type": "chat_response",
            "trace_id": trace_id,
            "answer": str(result.get("answer") or ""),
            "citations": [c.model_dump() if hasattr(c, "model_dump") else {"url": c.url, "snippet": c.snippet} for c in citations],
        }
    )
    return WidgetChatResponse(answer=str(result.get("answer") or ""), citations=citations)


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
    jobs = list_index_jobs_for_bot(bot_id)
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
            )
            for j in jobs
        ],
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
        return await start_index_for_bot(bot_id, payload.url)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/org/bots/{bot_id}/index/status")
async def v1_org_index_status(
    bot_id: str,
    url: str,
    org_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    resolved_org = _resolve_org_id(user, org_id)
    _assert_bot_org(bot_id, resolved_org)
    try:
        return get_index_status_for_bot(bot_id, url)
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
        return cancel_index_for_bot(bot_id, payload.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
