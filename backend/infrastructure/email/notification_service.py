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
    visitor_username: Optional[str] = None,
    visitor_name: Optional[str] = None,
    visitor_profile_pic_url: Optional[str] = None,
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
    if not chat_url:
        chat_url = _CHAT_LINKS.get(channel)
    dashboard_url = os.environ.get("DASHBOARD_URL", "").strip().rstrip("/")
    dash_link = f"{dashboard_url}/bots/{bot_id}/conversations?session={session_id}" if (dashboard_url and bot_id) else None

    # For Instagram: show visitor profile, make Instagram link primary
    profile_line = None
    if channel == "instagram" and (visitor_username or visitor_name):
        parts = []
        if visitor_name:
            parts.append(visitor_name)
        if visitor_username:
            parts.append(f"@{visitor_username}")
        profile_line = " · ".join(parts) if parts else None

    # Plain text body
    body_lines = [f"A visitor has requested human support via {channel_label}."]
    if profile_line:
        body_lines.extend(["", f"Visitor: {profile_line}"])
    if details:
        body_lines.extend(["", "What they said:", details])
    body_lines.append("")
    if chat_url:
        link_label = "Open Instagram messages to reply" if channel == "instagram" else f"Open {channel_label} to reply"
        body_lines.extend([f"{link_label}:", chat_url])
    if profile_line and visitor_username:
        body_lines.extend(["", f"View their profile: https://www.instagram.com/{visitor_username}"])
    if dash_link:
        body_lines.extend(["", f"View in dashboard: {dash_link}"])
    body = "\n".join(body_lines)

    # HTML version - Instagram primary: profile + Open messages button
    html_parts = [f"<p>A visitor has requested human support via {channel_label}.</p>"]
    if channel == "instagram" and (visitor_username or visitor_name or visitor_profile_pic_url):
        html_parts.append('<p style="display:flex;align-items:center;gap:12px;margin:16px 0;">')
        if visitor_profile_pic_url:
            html_parts.append(
                f'<img src="{html.escape(visitor_profile_pic_url)}" alt="" style="width:48px;height:48px;border-radius:50%;object-fit:cover;" />'
            )
        html_parts.append("<span>")
        if visitor_name:
            html_parts.append(f"<strong>{html.escape(visitor_name)}</strong> ")
        if visitor_username:
            profile_url = f"https://www.instagram.com/{visitor_username}"
            html_parts.append(f'<a href="{html.escape(profile_url)}" style="color:#0095f6;">@{html.escape(visitor_username)}</a>')
        html_parts.append("</span></p>")
    if details:
        safe_details = html.escape(details).replace("\n", "<br>")
        html_parts.append(f"<p><strong>What they said:</strong><br>{safe_details}</p>")
    if chat_url:
        btn_text = "Open Instagram messages" if channel == "instagram" else f"Open {channel_label} to reply"
        html_parts.append(
            f'<p style="margin-top:20px;">'
            f'<a href="{chat_url}" style="background:#0095f6;color:white;padding:12px 24px;text-decoration:none;border-radius:8px;display:inline-block;">{btn_text}</a>'
            f"</p>"
        )
    if profile_line and visitor_username:
        html_parts.append(
            f'<p><a href="https://www.instagram.com/{html.escape(visitor_username)}" style="color:#0095f6;">View their profile</a></p>'
        )
    if dash_link:
        html_parts.append(f'<p><a href="{dash_link}" style="color:#666;">View in dashboard</a></p>')
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
    visitor_username: Optional[str] = None,
    visitor_name: Optional[str] = None,
    visitor_profile_pic_url: Optional[str] = None,
) -> None:
    """
    Send escalation notification if configured for this channel.
    Call after create_escalation. bot must have escalation_config and display_name/bot_id.
    """
    bot_id = getattr(bot, "bot_id", "") or ""
    raw = getattr(bot, "escalation_config", None)
    if not raw or not str(raw).strip():
        logger.info(
            "Escalation email skipped: no escalation_config for bot_id=%s channel=%s",
            bot_id,
            channel,
        )
        return
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Escalation email skipped: invalid escalation_config JSON bot_id=%s", bot_id)
        return
    key = {"chat": "notify_website", "instagram": "notify_instagram", "line": "notify_line"}.get(channel)
    if not key:
        return
    legacy = bool(data.get("notify_enabled"))
    notify = bool(data.get(key)) if key in data else legacy
    if not notify:
        logger.info(
            "Escalation email skipped: notify_%s=False (or notify_enabled=False) for bot_id=%s",
            channel,
            bot_id,
        )
        return
    emails = parse_notification_emails(str(data.get("notification_emails") or ""))
    if not emails:
        logger.info(
            "Escalation email skipped: notification_emails empty for bot_id=%s channel=%s",
            bot_id,
            channel,
        )
        return
    bot_name = getattr(bot, "display_name", None) or getattr(bot, "bot_id", "Bot")
    send_escalation_notification(
        to_emails=emails,
        bot_name=bot_name,
        bot_id=bot_id,
        channel=channel,
        visitor_email=visitor_email,
        details=details,
        session_id=session_id,
        chat_url=chat_url,
        visitor_username=visitor_username,
        visitor_name=visitor_name,
        visitor_profile_pic_url=visitor_profile_pic_url,
    )
