from typing import Optional

import jwt
from fastapi import Header, HTTPException

from auth.jwt_auth import UserContext, build_user_context, is_super_admin, verify_token
from bot_registry import (
    add_membership,
    create_org,
    get_org_by_name,
    get_org_memberships,
    get_user_by_subject,
    upsert_user_from_claims,
)


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
        return f"Personal Org - {em}"
    sub = (subject or "").strip()
    if sub:
        return f"Personal Org - {sub}"
    return "Personal Org"


def get_current_user(authorization: Optional[str] = Header(default=None)) -> UserContext:
    token = _bearer_token(authorization)
    try:
        claims = verify_token(token)
        ctx = build_user_context(claims)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    user = get_user_by_subject(ctx.subject)
    if not user:
        user = upsert_user_from_claims(subject=ctx.subject, email=ctx.email)
    if ctx.email:
        # Link by email if a placeholder user exists.
        user = upsert_user_from_claims(subject=ctx.subject, email=ctx.email)

    ctx.user_id = user["user_id"]
    memberships = get_org_memberships(ctx.user_id)
    if not memberships:
        org_name = _org_name_for_user(ctx.email, ctx.subject)
        org = get_org_by_name(org_name)
        if org and org.get("status") == "active":
            org_id = org["org_id"]
        else:
            org_id = create_org(org_name)
        add_membership(org_id, ctx.user_id, "org_admin")
        memberships = get_org_memberships(ctx.user_id)
    ctx.org_ids = sorted(set(ctx.org_ids + [m["org_id"] for m in memberships]))
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
    memberships = get_org_memberships(ctx.user_id)
    role = next((m["role"] for m in memberships if m["org_id"] == org_id), "")
    if role not in {"org_admin", "owner"}:
        raise HTTPException(status_code=403, detail="Org admin access required")
    return ctx
