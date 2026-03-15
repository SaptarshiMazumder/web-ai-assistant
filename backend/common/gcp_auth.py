"""Shared GCP credential loading.

Supports two modes:
- Local dev: loads from JSON file via GOOGLE_APPLICATION_CREDENTIALS env var
- Cloud Run / GCE: uses Application Default Credentials (ADC) from metadata server
"""

import os
import logging
from typing import Tuple, Optional

import google.auth

logger = logging.getLogger(__name__)

_cached_credentials: Optional[Tuple] = None


def load_gcp_credentials() -> Tuple:
    """Load GCP credentials with file → ADC fallback.

    Returns (credentials, project_id) tuple.
    """
    global _cached_credentials
    if _cached_credentials is not None:
        return _cached_credentials

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if creds_path and os.path.exists(creds_path):
        creds, project = google.auth.load_credentials_from_file(creds_path)
        creds_type = "service_account_file"
        email = getattr(creds, "service_account_email", "N/A")
        logger.info("GCP auth: loaded from file (%s, project=%s, email=%s)", creds_type, project, email)
    else:
        creds, project = google.auth.default()
        creds_type = "adc"
        logger.info("GCP auth: using ADC (project=%s)", project)

    if not project:
        project = os.environ.get("PROJECT_ID", "")

    _cached_credentials = (creds, project)
    return creds, project


def has_gcp_credentials() -> bool:
    """Check if GCP credentials are available (file OR ADC)."""
    try:
        load_gcp_credentials()
        return True
    except Exception:
        return False
