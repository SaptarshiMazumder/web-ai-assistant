"""Send escalation notification emails via SMTP."""

import html
import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from common.config import config

logger = logging.getLogger(__name__)

_CHANNEL_LABELS = {
    "chat": "Website",
    "instagram": "Instagram",
    "line": "LINE",
}

# Links to open chat/inbox for each channel (client clicks to reply)
_CHAT_LINKS = {
    "instagram": "https://www.instagram.com/direct/inbox/",
    "line": "https://business.line.biz/",  # LINE Official Account Manager
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
    bot_id: str,
    channel: str,
    visitor_email: str,
    details: Optional[str] = None,
    session_id: str,
    chat_url: Optional[str] = None,
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

    # Add clickable link to open chat (use provided chat_url or fallback to channel default)
    if not chat_url:
        chat_url = _CHAT_LINKS.get(channel)
    dashboard_url = os.environ.get("DASHBOARD_URL", "").strip().rstrip("/")
    if chat_url:
        body_lines.extend(["", "Open inbox to reply:", chat_url])
    if dashboard_url and bot_id:
        body_lines.extend(["", "View in dashboard:", f"{dashboard_url}/bots/{bot_id}/conversations?session={session_id}"])

    body = "\n".join(body_lines)

    # HTML version with prominent button for the chat link
    html_parts = [
        f"<p>A visitor has requested human support via {channel_label}.</p>",
        f"<p><strong>Visitor contact:</strong> {html.escape(visitor_email)}</p>",
        f"<p><strong>Session ID:</strong> {html.escape(session_id)}</p>",
    ]
    if details:
        safe_details = html.escape(details).replace("\n", "<br>")
        html_parts.append(f"<p><strong>Details:</strong><br>{safe_details}</p>")
    if chat_url:
        html_parts.append(
            f'<p style="margin-top:20px;">'
            f'<a href="{chat_url}" style="background:#0095f6;color:white;padding:12px 24px;text-decoration:none;border-radius:8px;display:inline-block;">Open inbox to reply</a>'
            f"</p>"
        )
    if dashboard_url and bot_id:
        dash_link = f"{dashboard_url}/bots/{bot_id}/conversations?session={session_id}"
        html_parts.append(f'<p><a href="{dash_link}">View conversation in dashboard</a></p>')
    html_body = "".join(html_parts)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_FROM_EMAIL
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

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
    chat_url: Optional[str] = None,
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
    bot_id = getattr(bot, "bot_id", "") or ""
    send_escalation_notification(
        to_emails=emails,
        bot_name=bot_name,
        bot_id=bot_id,
        channel=channel,
        visitor_email=visitor_email,
        details=details,
        session_id=session_id,
        chat_url=chat_url,
    )
