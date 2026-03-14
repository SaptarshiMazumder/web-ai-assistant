import os
import threading
import time
from typing import Optional

import jwt
import requests
from fastapi import Header, HTTPException
from psycopg.errors import OperationalError

from application.auth.jwt_auth import UserContext, build_user_context, is_super_admin, verify_token
from common.di.deps import get_org_service, get_user_service


_AUTH_CONTEXT_CACHE_TTL_SEC = max(0, int((os.environ.get("AUTH_CONTEXT_CACHE_TTL_SEC") or "30").strip() or "30"))
_AUTH_CONTEXT_CACHE_MAX = max(64, int((os.environ.get("AUTH_CONTEXT_CACHE_MAX") or "2048").strip() or "2048"))
_AUTH_CONTEXT_CACHE: dict[str, tuple[float, UserContext]] = {}
_AUTH_CONTEXT_CACHE_LOCK = threading.Lock()


def _clone_user_context(ctx: UserContext) -> UserContext:
    return UserContext(
        user_id=ctx.user_id,
        subject=ctx.subject,
        email=ctx.email,
        org_ids=list(ctx.org_ids),
        roles=list(ctx.roles),
        claims=dict(ctx.claims),
    )


def _cache_get_user_context(token: str) -> Optional[UserContext]:
    if _AUTH_CONTEXT_CACHE_TTL_SEC <= 0:
        return None
    now = time.time()
    with _AUTH_CONTEXT_CACHE_LOCK:
        entry = _AUTH_CONTEXT_CACHE.get(token)
        if not entry:
            return None
        expires_at, cached = entry
        if expires_at <= now:
            _AUTH_CONTEXT_CACHE.pop(token, None)
            return None
        return _clone_user_context(cached)


def _cache_put_user_context(token: str, ctx: UserContext) -> None:
    if _AUTH_CONTEXT_CACHE_TTL_SEC <= 0:
        return
    now = time.time()
    expiry = now + _AUTH_CONTEXT_CACHE_TTL_SEC
    exp_claim = ctx.claims.get("exp")
    if isinstance(exp_claim, (int, float)):
        expiry = min(expiry, float(exp_claim) - 5.0)
    if expiry <= now:
        return
    with _AUTH_CONTEXT_CACHE_LOCK:
        if len(_AUTH_CONTEXT_CACHE) >= _AUTH_CONTEXT_CACHE_MAX:
            # Drop oldest by expiry to keep inserts O(n) only at capacity boundary.
            oldest_key = min(_AUTH_CONTEXT_CACHE, key=lambda k: _AUTH_CONTEXT_CACHE[k][0])
            _AUTH_CONTEXT_CACHE.pop(oldest_key, None)
        _AUTH_CONTEXT_CACHE[token] = (expiry, _clone_user_context(ctx))


def _bearer_token(authorization: Optional[str]) -> str:
    auth = (authorization or "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")
    return auth.split(" ", 1)[1].strip()


_PUBLIC_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "live.com",
    "msn.com",
    "yandex.com",
    "yandex.ru",
    "qq.com",
}


def _email_domain(email: str) -> str:
    em = (email or "").strip().lower()
    if "@" not in em:
        return ""
    return em.split("@", 1)[1].strip()


def _org_name_for_user(email: str, subject: str) -> str:
    domain = _email_domain(email)
    if domain and domain not in _PUBLIC_EMAIL_DOMAINS:
        return domain
    em = (email or "").strip().lower()
    if em:
        local_part = em.split("@", 1)[0].strip() if "@" in em else em
        return f"Personal Org - {local_part}"
    sub = (subject or "").strip()
    if sub:
        return f"Personal Org - {sub}"
    return "Personal Org"


def _extract_name_parts(claims: dict) -> tuple[Optional[str], Optional[str]]:
    given = str(claims.get("given_name") or "").strip()
    family = str(claims.get("family_name") or "").strip()
    if given or family:
        return (given or None, family or None)
    full = str(claims.get("name") or "").strip()
    if not full:
        return (None, None)
    parts = [p for p in full.replace(",", " ").split(" ") if p]
    if not parts:
        return (None, None)
    if len(parts) == 1:
        return (parts[0], None)
    return (parts[0], " ".join(parts[1:]))


def get_current_user(authorization: Optional[str] = Header(default=None)) -> UserContext:
    token = _bearer_token(authorization)
    cached_ctx = _cache_get_user_context(token)
    if cached_ctx is not None:
        return cached_ctx
    try:
        claims = verify_token(token)
        ctx = build_user_context(claims)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Authentication provider unavailable: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Authentication configuration error: {exc}")

    user_service = get_user_service()
    org_service = get_org_service()

    first_name, last_name = _extract_name_parts(claims)

    for attempt in range(2):
        try:
            user = user_service.get_user_by_subject(ctx.subject)
            if not user:
                user = user_service.upsert_user_from_claims(
                    subject=ctx.subject,
                    email=ctx.email,
                    first_name=first_name,
                    last_name=last_name,
                )
            elif ctx.email and (
                (user.email or "").strip().lower() != (ctx.email or "").strip().lower()
                or (first_name and (user.first_name or "") != first_name)
                or (last_name and (user.last_name or "") != last_name)
            ):
                user = user_service.upsert_user_from_claims(
                    subject=ctx.subject,
                    email=ctx.email,
                    first_name=first_name,
                    last_name=last_name,
                )

            ctx.user_id = user.user_id
            memberships = org_service.get_org_memberships(ctx.user_id)
            if not memberships:
                org_name = _org_name_for_user(ctx.email, ctx.subject)
                org = org_service.get_org_by_name(org_name)
                if org and org.status == "active":
                    org_id = org.org_id
                else:
                    org_id = org_service.create_org(org_name)
                org_service.add_membership(org_id, ctx.user_id, "org_admin")
                memberships = org_service.get_org_memberships(ctx.user_id)
            ctx.org_ids = sorted(set(ctx.org_ids + [m["org_id"] for m in memberships]))
            break
        except OperationalError as exc:
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    _cache_put_user_context(token, ctx)
    return ctx


def require_super_admin(authorization: Optional[str] = Header(default=None)) -> UserContext:
    ctx = get_current_user(authorization)
    if not is_super_admin(ctx.claims):
        raise HTTPException(status_code=403, detail="Super admin access required")
    return ctx


def require_org_member(
    org_id: str,
    authorization: Optional[str] = Header(default=None),
) -> UserContext:
    ctx = get_current_user(authorization)
    if org_id not in ctx.org_ids and not is_super_admin(ctx.claims):
        raise HTTPException(status_code=403, detail="Org membership required")
    return ctx


def require_org_admin(
    org_id: str,
    authorization: Optional[str] = Header(default=None),
) -> UserContext:
    ctx = get_current_user(authorization)
    if is_super_admin(ctx.claims):
        return ctx
    org_service = get_org_service()
    memberships = org_service.get_org_memberships(ctx.user_id)
    role = next((m["role"] for m in memberships if m["org_id"] == org_id), "")
    if role not in {"org_admin", "owner"}:
        raise HTTPException(status_code=403, detail="Org admin access required")
    return ctx
