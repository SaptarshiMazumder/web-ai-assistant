"""Send escalation notification emails via SMTP."""

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from common import config

logger = logging.getLogger(__name__)

_CHANNEL_LABELS = {
    "chat": "Website",
    "instagram": "Instagram",
    "line": "LINE",
}


def parse_notification_emails(raw: str) -> List[str]:
    """Parse semicolon- or comma-separated emails, strip whitespace."""
    if not raw or not raw.strip():
        return []
    parts = raw.replace(",", ";").split(";")
    return [p.strip().lower() for p in parts if p.strip() and "@" in p.strip()]


def send_escalation_notification(
    *,
    to_emails: List[str],
    bot_name: str,
    channel: str,
    visitor_email: str,
    details: Optional[str] = None,
    session_id: str,
) -> bool:
    """
    Send an email notification when a visitor escalates to human support.
    Returns True if sent successfully, False otherwise.
    """
    if not to_emails:
        return False
    if not config.SMTP_HOST or not config.SMTP_FROM_EMAIL:
        logger.warning("Escalation email skipped: SMTP_HOST or SMTP_FROM_EMAIL not configured")
        return False

    channel_label = _CHANNEL_LABELS.get(channel, channel)
    subject = f"[{bot_name}] Human support request via {channel_label}"
    body_lines = [
        f"A visitor has requested human support via {channel_label}.",
        "",
        f"Visitor contact: {visitor_email}",
        f"Session ID: {session_id}",
    ]
    if details:
        body_lines.extend(["", "Details:", details])
    body = "\n".join(body_lines)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_FROM_EMAIL
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            if config.SMTP_USE_TLS:
                smtp.starttls()
            if config.SMTP_USER and config.SMTP_PASSWORD:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.sendmail(config.SMTP_FROM_EMAIL, to_emails, msg.as_string())
        logger.info("Escalation notification sent to %s for session %s", to_emails, session_id)
        return True
    except Exception as e:
        logger.exception("Failed to send escalation email: %s", e)
        return False


def maybe_send_escalation_email(
    bot,
    session_id: str,
    channel: str,
    visitor_email: str,
    details: Optional[str] = None,
) -> None:
    """
    Send escalation notification if configured for this channel.
    Call after create_escalation. bot must have escalation_config and display_name/bot_id.
    """
    raw = getattr(bot, "escalation_config", None)
    if not raw or not str(raw).strip():
        return
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return
    key = {"chat": "notify_website", "instagram": "notify_instagram", "line": "notify_line"}.get(channel)
    if not key:
        return
    legacy = bool(data.get("notify_enabled"))
    notify = bool(data.get(key)) if key in data else legacy
    if not notify:
        return
    emails = parse_notification_emails(str(data.get("notification_emails") or ""))
    if not emails:
        return
    bot_name = getattr(bot, "display_name", None) or getattr(bot, "bot_id", "Bot")
    send_escalation_notification(
        to_emails=emails,
        bot_name=bot_name,
        channel=channel,
        visitor_email=visitor_email,
        details=details,
        session_id=session_id,
    )
