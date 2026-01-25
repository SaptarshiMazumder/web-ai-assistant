import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jwt
import requests

from config import config


@dataclass
class UserContext:
    user_id: str
    subject: str
    email: str
    org_ids: List[str]
    roles: List[str]
    claims: Dict[str, Any]


_JWKS_CACHE: Dict[str, Any] = {"expires_at": 0, "jwks": None}


def _jwks_url() -> str:
    issuer = (config.AUTH_ISSUER or "").rstrip("/")
    if not issuer:
        raise RuntimeError("AUTH_ISSUER is not configured")
    return config.AUTH_JWKS_URL or f"{issuer}/.well-known/jwks.json"


def _get_jwks() -> Dict[str, Any]:
    now = int(time.time())
    if _JWKS_CACHE["jwks"] and _JWKS_CACHE["expires_at"] > now:
        return _JWKS_CACHE["jwks"]
    url = _jwks_url()
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    jwks = resp.json()
    _JWKS_CACHE["jwks"] = jwks
    _JWKS_CACHE["expires_at"] = now + 3600
    return jwks


def _resolve_key(token: str) -> Any:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
    raise jwt.InvalidKeyError("No matching JWKS key")


def verify_token(token: str) -> Dict[str, Any]:
    key = _resolve_key(token)
    issuer = (config.AUTH_ISSUER or "").strip()
    audience = (config.AUTH_AUDIENCE or "").strip()
    if not audience:
        raise RuntimeError("AUTH_AUDIENCE is not configured")
    if not issuer:
        raise RuntimeError("AUTH_ISSUER is not configured")
    claims = jwt.decode(
        token,
        key=key,
        algorithms=["RS256"],
        audience=audience,
        options={"verify_iss": False},
    )
    token_iss = str(claims.get("iss") or "").strip()
    if not token_iss:
        raise jwt.InvalidIssuerError("Missing iss claim")
    if token_iss.rstrip("/") != issuer.rstrip("/"):
        raise jwt.InvalidIssuerError(f"Invalid issuer: {token_iss}")
    return claims


def extract_roles(claims: Dict[str, Any]) -> List[str]:
    roles = []
    for key in ("roles", "role", "permissions"):
        val = claims.get(key)
        if isinstance(val, list):
            roles.extend([str(v) for v in val])
        elif isinstance(val, str):
            roles.append(val)
    # Support custom namespaced claim
    ns_key = (config.AUTH_ROLES_CLAIM or "").strip()
    if ns_key:
        val = claims.get(ns_key)
        if isinstance(val, list):
            roles.extend([str(v) for v in val])
    return sorted(set(roles))


def extract_org_ids(claims: Dict[str, Any]) -> List[str]:
    org_ids = []
    for key in ("org_id", "org", "organization_id"):
        val = claims.get(key)
        if isinstance(val, list):
            org_ids.extend([str(v) for v in val])
        elif isinstance(val, str):
            org_ids.append(val)
    ns_key = (config.AUTH_ORG_CLAIM or "").strip()
    if ns_key:
        val = claims.get(ns_key)
        if isinstance(val, list):
            org_ids.extend([str(v) for v in val])
        elif isinstance(val, str):
            org_ids.append(val)
    return sorted(set([o for o in org_ids if o]))


def is_super_admin(claims: Dict[str, Any]) -> bool:
    roles = extract_roles(claims)
    if "super_admin" in roles or "owner" in roles:
        return True
    emails = [e.strip().lower() for e in (config.SUPER_ADMIN_EMAILS or "").split(",") if e.strip()]
    email = str(claims.get("email") or "").strip().lower()
    return bool(email and email in emails)


def build_user_context(claims: Dict[str, Any]) -> UserContext:
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise jwt.InvalidTokenError("Missing sub claim")
    email = str(claims.get("email") or "").strip()
    return UserContext(
        user_id="",
        subject=subject,
        email=email,
        org_ids=extract_org_ids(claims),
        roles=extract_roles(claims),
        claims=claims,
    )
