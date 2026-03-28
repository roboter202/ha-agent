"""
IMAP email client.
Connects to any IMAP server (Gmail, self-hosted Mailcow/Dovecot, etc.)
and provides async helpers for:
  - Unread count + inbox summary
  - Recent email listing (subject, sender, date, snippet)
  - Full message fetch for summarisation
  - Search by sender / subject / date range

Credentials live in config/settings.yaml under `email:` or env vars.
"""

from __future__ import annotations

import asyncio
import email
import email.header
import email.utils
import re
from datetime import date, datetime, timedelta
from email.message import Message
from typing import Any

import aioimaplib
import structlog

from core.config import get_settings

log = structlog.get_logger(__name__)


def _decode_header(raw: str | bytes) -> str:
    """Safely decode an RFC-2047-encoded header value."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = email.header.decode_header(raw)
    decoded = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            decoded.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(chunk)
    return "".join(decoded)


def _get_text_body(msg: Message) -> str:
    """Extract plain-text body from a (possibly multi-part) email."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                return payload.decode(charset, errors="replace") if payload else ""
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        return payload.decode(charset, errors="replace") if payload else ""
    return ""


class EmailClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._host = cfg.get("email", "host", default="")
        self._port = int(cfg.get("email", "port", default=993))
        self._user = cfg.get("email", "username", default="")
        self._password = cfg.get("email", "password", default="")
        self._ssl = cfg.get("email", "ssl", default=True)
        self._default_folder = cfg.get("email", "folder", default="INBOX")

    def _is_configured(self) -> bool:
        return bool(self._host and self._user and self._password)

    async def _connect(self) -> aioimaplib.IMAP4_SSL | aioimaplib.IMAP4:
        if self._ssl:
            client = aioimaplib.IMAP4_SSL(host=self._host, port=self._port)
        else:
            client = aioimaplib.IMAP4(host=self._host, port=self._port)
        await client.wait_hello_from_server()
        await client.login(self._user, self._password)
        return client

    # ── Public API ────────────────────────────────────────────────────────

    async def get_unread_count(self, folder: str | None = None) -> int:
        """Return number of unseen messages in the folder."""
        if not self._is_configured():
            return -1
        folder = folder or self._default_folder
        client = await self._connect()
        try:
            await client.select(folder, readonly=True)
            status, lines = await client.search("UNSEEN")
            if status == "OK" and lines and lines[0]:
                uids = lines[0].split()
                return len(uids)
            return 0
        finally:
            await client.logout()

    async def list_recent(
        self,
        folder: str | None = None,
        limit: int = 10,
        only_unread: bool = False,
        since: date | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of recent messages with metadata."""
        if not self._is_configured():
            return []
        folder = folder or self._default_folder
        since = since or (date.today() - timedelta(days=7))
        since_str = since.strftime("%d-%b-%Y")

        client = await self._connect()
        try:
            await client.select(folder, readonly=True)
            criteria = f"SINCE {since_str}"
            if only_unread:
                criteria = f"UNSEEN SINCE {since_str}"

            status, lines = await client.search(criteria)
            if status != "OK" or not lines or not lines[0]:
                return []

            uids = lines[0].split()[-limit:]  # most recent N
            messages = []

            for uid in reversed(uids):
                fetch_status, fetch_data = await client.fetch(
                    uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                if fetch_status != "OK":
                    continue
                raw_header = b""
                for part in fetch_data:
                    if isinstance(part, bytes) and part.strip():
                        raw_header = part
                        break

                msg = email.message_from_bytes(raw_header)
                messages.append(
                    {
                        "uid": uid.decode() if isinstance(uid, bytes) else uid,
                        "from": _decode_header(msg.get("From", "")),
                        "subject": _decode_header(msg.get("Subject", "(no subject)")),
                        "date": msg.get("Date", ""),
                    }
                )

            return messages
        finally:
            await client.logout()

    async def fetch_message(self, uid: str, folder: str | None = None) -> dict[str, Any]:
        """Fetch full message (headers + body snippet) by UID."""
        if not self._is_configured():
            return {}
        folder = folder or self._default_folder
        client = await self._connect()
        try:
            await client.select(folder, readonly=True)
            status, fetch_data = await client.fetch(uid, "(RFC822)")
            if status != "OK":
                return {}

            raw = b""
            for part in fetch_data:
                if isinstance(part, bytes) and len(part) > 100:
                    raw = part
                    break

            msg = email.message_from_bytes(raw)
            body = _get_text_body(msg)
            return {
                "uid": uid,
                "from": _decode_header(msg.get("From", "")),
                "to": _decode_header(msg.get("To", "")),
                "subject": _decode_header(msg.get("Subject", "")),
                "date": msg.get("Date", ""),
                "body_snippet": body[:1500],  # cap for LLM context
            }
        finally:
            await client.logout()

    async def search_messages(
        self,
        sender: str | None = None,
        subject_contains: str | None = None,
        since: date | None = None,
        limit: int = 5,
        folder: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search emails by sender / subject / date."""
        if not self._is_configured():
            return []
        folder = folder or self._default_folder
        client = await self._connect()
        try:
            await client.select(folder, readonly=True)
            parts = []
            if sender:
                parts.append(f'FROM "{sender}"')
            if subject_contains:
                parts.append(f'SUBJECT "{subject_contains}"')
            if since:
                parts.append(f"SINCE {since.strftime('%d-%b-%Y')}")
            criteria = " ".join(parts) or "ALL"

            status, lines = await client.search(criteria)
            if status != "OK" or not lines or not lines[0]:
                return []

            uids = lines[0].split()[-limit:]
            results = []
            for uid in reversed(uids):
                fetch_status, fetch_data = await client.fetch(
                    uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                if fetch_status != "OK":
                    continue
                raw_header = b""
                for part in fetch_data:
                    if isinstance(part, bytes) and part.strip():
                        raw_header = part
                        break
                msg = email.message_from_bytes(raw_header)
                results.append(
                    {
                        "uid": uid.decode() if isinstance(uid, bytes) else uid,
                        "from": _decode_header(msg.get("From", "")),
                        "subject": _decode_header(msg.get("Subject", "")),
                        "date": msg.get("Date", ""),
                    }
                )
            return results
        finally:
            await client.logout()

    async def get_inbox_summary(self) -> dict[str, Any]:
        """Return a quick inbox overview: unread count + last 5 subjects."""
        unread = await self.get_unread_count()
        recent = await self.list_recent(limit=5, only_unread=True)
        return {
            "unread_count": unread,
            "recent_unread": [
                {"from": m["from"], "subject": m["subject"], "date": m["date"]}
                for m in recent
            ],
        }


_client: EmailClient | None = None


def get_email() -> EmailClient:
    global _client
    if _client is None:
        _client = EmailClient()
    return _client
