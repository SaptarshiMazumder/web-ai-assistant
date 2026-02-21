import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import jwt
import requests

from common.config import config


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
    # Allow JWKS to be provided directly via env var (useful when DNS is restricted)
    static_jwks = (config.AUTH_JWKS_JSON or "").strip()
    if static_jwks:
        if not _JWKS_CACHE["jwks"]:
            try:
                _JWKS_CACHE["jwks"] = json.loads(static_jwks)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"AUTH_JWKS_JSON is invalid JSON: {exc}") from exc
            _JWKS_CACHE["expires_at"] = int(time.time()) + 86400 * 365
        return _JWKS_CACHE["jwks"]

    now = int(time.time())
    cached_jwks = _JWKS_CACHE["jwks"]
    if cached_jwks and _JWKS_CACHE["expires_at"] > now:
        return cached_jwks
    url = _jwks_url()
    last_err: Exception = RuntimeError("JWKS fetch failed")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            jwks = resp.json()
            _JWKS_CACHE["jwks"] = jwks
            _JWKS_CACHE["expires_at"] = now + 3600
            return jwks
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    # If refresh failed, use last known JWKS as a temporary fallback.
    if cached_jwks:
        _JWKS_CACHE["expires_at"] = now + 300
        return cached_jwks
    raise last_err


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
