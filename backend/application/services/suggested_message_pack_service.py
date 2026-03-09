from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from google import genai

from common.config import config
from domain.entities import SuggestedMessageFastPathResult, SuggestedMessagePack
from domain.platform_profiles import (
    get_action_destination_url,
    get_reservation_config_from_widget,
    get_suggested_messages_by_language_for_widget,
    get_suggested_messages_for_widget,
)
from infrastructure.clients.rag_client import (
    GENAI_LOCATION,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    PROJECT_ID,
    sanitize_answer_citations,
    synthesize_with_evidence,
    synthesize_with_evidence_stream,
)
from infrastructure.db.repositories import (
    PostgresBotAssetRepository,
    PostgresBotRepository,
    PostgresIndexJobRepository,
    PostgresSuggestedMessagePackRepository,
)

logger = logging.getLogger(__name__)

_FAST_PATH_MODEL = (
    os.environ.get("SUGGESTED_MESSAGE_FAST_PATH_MODEL")
    or os.environ.get("VERTEX_RAG_MODEL")
    or DEFAULT_MODEL_NAME
).strip()
_EVIDENCE_LIMIT = 4
_SNIPPET_MAX_CHARS = 900
_PACKABLE_INDEX_STAGES = {"done", "import_submitted", "prompt_queued", "prompt_generating"}
_BINDING_KEYS = ("fastPathBinding", "binding")
_GENERIC_PROMPT_SEEDS = {
    "ask a question",
    "ask me anything",
    "question",
    "質問する",
    "質問",
}
_HTTP_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_lang(lang: Optional[str]) -> str:
    raw = (lang or "en").strip().lower()
    return "ja" if raw in {"ja", "jp"} else "en"


def _normalize_http_url(url: Any) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    try:
        parsed = urlsplit(value)
    except Exception:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    return parsed._replace(path=path, fragment="").geturl()


def _url_keys(url: str) -> Tuple[Optional[Tuple[str, str, str, str]], Optional[Tuple[str, str, str]]]:
    normalized = _normalize_http_url(url)
    if not normalized:
        return None, None
    parsed = urlsplit(normalized)
    host = (parsed.netloc or "").lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return (parsed.scheme.lower(), host, path, parsed.query or ""), (parsed.scheme.lower(), host, path)


def _friendly_label_from_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        path = (parsed.path or "").strip("/")
        if not path:
            return (parsed.netloc or "this page").strip()
        leaf = path.split("/")[-1].replace("-", " ").replace("_", " ").strip()
        return leaf.title() if leaf else (parsed.netloc or "this page").strip()
    except Exception:
        return "this page"


def _extract_doc_title(content: str, url: str) -> str:
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("source url:"):
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()[:120]
        return line[:120]
    return _friendly_label_from_url(url)


def _extract_doc_snippet(content: str, *, limit: int = _SNIPPET_MAX_CHARS) -> str:
    text = str(content or "")
    if text.lower().startswith("source url:"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]*\)", r"\1", text)
    blocks = re.split(r"\n\s*\n", text)
    chosen: List[str] = []
    used = 0
    for block in blocks:
        cleaned = re.sub(r"^#+\s*", "", block.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if len(cleaned) > 320:
            cleaned = cleaned[:320].rstrip() + "..."
        chosen.append(cleaned)
        used += len(cleaned)
        if len(chosen) >= 3 or used >= limit:
            break
    snippet = "\n\n".join(chosen).strip()
    return snippet[:limit].strip()


def _dedupe_urls(urls: Iterable[Any]) -> List[str]:
    seen = set()
    output: List[str] = []
    for raw in urls:
        normalized = _normalize_http_url(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _extract_action_link_target(pack: SuggestedMessagePack) -> Tuple[str, str]:
    for raw in pack.link_targets or []:
        if not isinstance(raw, dict):
            continue
        url = _normalize_http_url(raw.get("url"))
        if not url:
            continue
        label = str(raw.get("label") or raw.get("title") or "").strip()
        return url, label
    for raw in pack.citations or []:
        if not isinstance(raw, dict):
            continue
        url = _normalize_http_url(raw.get("url"))
        if not url:
            continue
        label = str(raw.get("label") or raw.get("title") or "").strip()
        return url, label
    for raw in pack.source_urls or []:
        url = _normalize_http_url(raw)
        if url:
            return url, ""
    return "", ""


def _ensure_action_link_visible(answer: str, *, pack: SuggestedMessagePack) -> str:
    text = str(answer or "").strip()
    if str(pack.pack_mode or "").strip().lower() != "action_link":
        return text
    if _HTTP_URL_RE.search(text):
        return text
    url, label = _extract_action_link_target(pack)
    if not url:
        return text
    link_text = f"{label}: {url}" if label and label != url else url
    if not text:
        return link_text
    return f"{text}\n{link_text}"


def _load_docs_from_gcs_prefix(gcs_prefix: str) -> List[Dict[str, Any]]:
    if not gcs_prefix:
        return []
    bucket_raw = (config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")).strip()
    if not bucket_raw:
        return []
    bucket_name = bucket_raw.strip("/").split("/", 1)[0]
    try:
        from google.cloud import storage
    except Exception:
        return []
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    documents: List[Dict[str, Any]] = []
    for blob in bucket.list_blobs(prefix=gcs_prefix):
        if not blob.name.endswith(".md"):
            continue
        try:
            content = blob.download_as_text(encoding="utf-8")
        except Exception:
            continue
        url = ""
        if content.startswith("Source URL:"):
            first_line = content.split("\n", 1)[0]
            url = first_line.replace("Source URL:", "").strip()
        documents.append({"url": url, "content": content, "blob_name": blob.name})
    return documents


def _index_docs(documents: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        url = _normalize_http_url(doc.get("url"))
        exact_key, base_key = _url_keys(url)
        if not exact_key and not base_key:
            continue
        indexed_doc = {
            "url": url,
            "title": _extract_doc_title(doc.get("content", ""), url),
            "snippet": _extract_doc_snippet(doc.get("content", "")),
        }
        if exact_key:
            indexed[json.dumps(exact_key)] = indexed_doc
        if base_key:
            indexed.setdefault(json.dumps(base_key), indexed_doc)
    return indexed


def _infer_binding(item: Dict[str, Any]) -> Optional[str]:
    for key in _BINDING_KEYS:
        value = str(item.get(key) or "").strip().lower()
        if value:
            return value
    raw_type = str(item.get("type") or "").strip().lower()
    if raw_type == "show_menu":
        return "menu"
    if raw_type == "escalate":
        return "support"
    urls = item.get("urls")
    if isinstance(urls, list) and any(str(url).strip() for url in urls):
        return "explicit_urls"
    prompt_seed = str(item.get("prompt") or item.get("message") or item.get("label") or "").strip()
    if raw_type == "ai_response" and prompt_seed and prompt_seed.lower() not in _GENERIC_PROMPT_SEEDS:
        return "semantic_prompt"
    return None


class SuggestedMessagePackBuilderService:
    def __init__(
        self,
        *,
        bot_repo: Optional[PostgresBotRepository] = None,
        index_repo: Optional[PostgresIndexJobRepository] = None,
        asset_repo: Optional[PostgresBotAssetRepository] = None,
        pack_repo: Optional[PostgresSuggestedMessagePackRepository] = None,
    ) -> None:
        self._bot_repo = bot_repo or PostgresBotRepository()
        self._index_repo = index_repo or PostgresIndexJobRepository()
        self._asset_repo = asset_repo or PostgresBotAssetRepository()
        self._pack_repo = pack_repo or PostgresSuggestedMessagePackRepository()

    def rebuild_for_bot(self, bot_id: str, *, widget_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        bot = self._bot_repo.get_bot(bot_id)
        if not bot:
            return {"langs": 0, "packs": 0}
        cfg = dict(widget_config) if isinstance(widget_config, dict) else {}
        if not cfg and getattr(bot, "widget_config", None):
            try:
                cfg = json.loads(bot.widget_config or "{}")
            except (TypeError, ValueError):
                cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}

        latest_gcs_prefix = self._latest_gcs_prefix(bot_id)
        docs_by_key = _index_docs(_load_docs_from_gcs_prefix(latest_gcs_prefix)) if latest_gcs_prefix else {}
        assets = self._asset_repo.list_assets_for_bot(bot_id, active_only=True, asset_type="menu_item")
        menu_source_urls = _dedupe_urls(
            (
                ((asset.metadata or {}).get("source_url") if isinstance(getattr(asset, "metadata", None), dict) else None)
                or getattr(asset, "link_url", None)
                for asset in assets
            )
        )

        by_lang = get_suggested_messages_by_language_for_widget(cfg)
        total = 0
        for lang, items in by_lang.items():
            packs: List[SuggestedMessagePack] = []
            for item in items or []:
                pack = self._build_pack(
                    bot_id=bot.bot_id,
                    org_id=bot.org_id,
                    lang=_normalize_lang(lang),
                    item=item,
                    widget_config=cfg,
                    docs_by_key=docs_by_key,
                    latest_gcs_prefix=latest_gcs_prefix,
                    menu_source_urls=menu_source_urls,
                )
                if pack is not None:
                    packs.append(pack)
            self._pack_repo.replace_for_bot_lang(bot_id=bot.bot_id, lang=_normalize_lang(lang), packs=packs)
            total += len(packs)
        return {"langs": len(by_lang), "packs": total}

    def compute_expected_version_hash(
        self,
        *,
        bot_id: str,
        widget_config: Dict[str, Any],
        lang: str,
        suggested_message_id: str,
    ) -> Optional[str]:
        normalized_lang = _normalize_lang(lang)
        item = next(
            (
                candidate
                for candidate in get_suggested_messages_for_widget(widget_config, lang=normalized_lang)
                if str(candidate.get("id") or "").strip() == str(suggested_message_id or "").strip()
            ),
            None,
        )
        if not item:
            return None
        binding = _infer_binding(item)
        if not binding:
            return None
        latest_gcs_prefix = self._latest_gcs_prefix(bot_id)
        docs_by_key = _index_docs(_load_docs_from_gcs_prefix(latest_gcs_prefix)) if latest_gcs_prefix else {}
        assets = self._asset_repo.list_assets_for_bot(bot_id, active_only=True, asset_type="menu_item")
        menu_source_urls = _dedupe_urls(
            (
                ((asset.metadata or {}).get("source_url") if isinstance(getattr(asset, "metadata", None), dict) else None)
                or getattr(asset, "link_url", None)
                for asset in assets
            )
        )
        payload = self._resolve_binding_payload(
            bot_id=bot_id,
            binding=binding,
            lang=normalized_lang,
            item=item,
            widget_config=widget_config,
            docs_by_key=docs_by_key,
            menu_source_urls=menu_source_urls,
            latest_gcs_prefix=latest_gcs_prefix,
        )
        if payload is None:
            return None
        return self._compute_version_hash(
            lang=normalized_lang,
            item=item,
            binding=binding,
            payload=payload,
            latest_gcs_prefix=latest_gcs_prefix,
        )

    def _latest_gcs_prefix(self, bot_id: str) -> str:
        jobs = self._index_repo.list_jobs_for_bot(bot_id)
        for job in jobs:
            stage = str(getattr(job, "stage", "") or "").strip().lower()
            prefix = str(getattr(job, "gcs_prefix", "") or "").strip()
            if prefix and (not stage or stage in _PACKABLE_INDEX_STAGES):
                return prefix
        return ""

    def _build_pack(
        self,
        *,
        bot_id: str,
        org_id: str,
        lang: str,
        item: Dict[str, Any],
        widget_config: Dict[str, Any],
        docs_by_key: Dict[str, Dict[str, Any]],
        latest_gcs_prefix: str,
        menu_source_urls: Sequence[str],
    ) -> Optional[SuggestedMessagePack]:
        suggested_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()
        prompt = str(item.get("prompt") or item.get("message") or label).strip()
        if not suggested_id or not label or not prompt:
            return None
        binding = _infer_binding(item)
        if not binding:
            return None
        payload = self._resolve_binding_payload(
            bot_id=bot_id,
            binding=binding,
            lang=lang,
            item=item,
            widget_config=widget_config,
            docs_by_key=docs_by_key,
            menu_source_urls=menu_source_urls,
            latest_gcs_prefix=latest_gcs_prefix,
        )
        if payload is None:
            return None
        now = _utc_now()
        return SuggestedMessagePack(
            pack_id=f"smp_{uuid.uuid4().hex}",
            bot_id=bot_id,
            org_id=org_id,
            lang=lang,
            suggested_message_id=suggested_id,
            label=label,
            prompt=prompt,
            pack_mode=payload["pack_mode"],
            status="ready",
            version_hash=self._compute_version_hash(
                lang=lang,
                item=item,
                binding=binding,
                payload=payload,
                latest_gcs_prefix=latest_gcs_prefix,
            ),
            source_urls=list(payload["source_urls"]),
            evidence_snippets=list(payload["evidence_snippets"]),
            link_targets=list(payload["link_targets"]),
            instruction=str(payload.get("instruction") or ""),
            citations=list(payload["citations"]),
            error=None,
            created_at=now,
            updated_at=now,
        )

    def _resolve_binding_payload(
        self,
        *,
        bot_id: str,
        binding: str,
        lang: str,
        item: Dict[str, Any],
        widget_config: Dict[str, Any],
        docs_by_key: Dict[str, Dict[str, Any]],
        menu_source_urls: Sequence[str],
        latest_gcs_prefix: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_binding = (binding or "").strip().lower()
        if normalized_binding == "reservation":
            reservation_cfg = get_reservation_config_from_widget(widget_config, lang=lang)
            if not reservation_cfg:
                return None
            target_url = _normalize_http_url(reservation_cfg.get("url"))
            if not target_url:
                return None
            label = str(reservation_cfg.get("link_label") or item.get("label") or "Reservation").strip()
            snippet = f"Official reservation link: {target_url}"
            return {
                "pack_mode": "action_link",
                "source_urls": [target_url],
                "evidence_snippets": [{"url": target_url, "snippet": snippet, "title": label}],
                "link_targets": [{"url": target_url, "label": label, "source_kind": "reservation"}],
                "instruction": str(reservation_cfg.get("instruction") or "").strip(),
                "citations": [{"url": target_url, "snippet": snippet, "title": label}],
            }
        if normalized_binding == "menu":
            urls = _dedupe_urls(
                [
                    *(item.get("urls") or []),
                    get_action_destination_url(widget_config, "menu"),
                    *menu_source_urls,
                ]
            )
            return self._build_url_payload(
                urls=urls,
                docs_by_key=docs_by_key,
                fallback_label=str(item.get("label") or "Menu").strip(),
                instruction="Use the provided menu evidence and menu links only. Do not invent items that are not in the pack.",
            )
        if normalized_binding in {"explicit_urls", "page_urls", "page_evidence"}:
            urls = _dedupe_urls(item.get("urls") or [])
            return self._build_url_payload(
                urls=urls,
                docs_by_key=docs_by_key,
                fallback_label=str(item.get("label") or "this page").strip(),
                instruction="Answer only from the provided page evidence and links. If the pack does not confirm the answer, say so.",
            )
        if normalized_binding == "support":
            support_url = _normalize_http_url(
                get_action_destination_url(widget_config, "support")
                or get_action_destination_url(widget_config, "contact")
            )
            if not support_url:
                return None
            label = str(item.get("label") or "Support").strip()
            snippet = f"Support link: {support_url}"
            return {
                "pack_mode": "action_link",
                "source_urls": [support_url],
                "evidence_snippets": [{"url": support_url, "snippet": snippet, "title": label}],
                "link_targets": [{"url": support_url, "label": label, "source_kind": "support"}],
                "instruction": "Use the configured support link only. Do not invent phone numbers, email addresses, or hours.",
                "citations": [{"url": support_url, "snippet": snippet, "title": label}],
            }
        if normalized_binding == "semantic_prompt":
            prompt = str(item.get("prompt") or item.get("message") or item.get("label") or "").strip()
            label = str(item.get("label") or prompt or "this topic").strip()
            return self._build_prompt_payload(
                bot_id=bot_id,
                prompt=prompt,
                label=label,
                latest_gcs_prefix=latest_gcs_prefix,
            )
        return None

    def _build_prompt_payload(
        self,
        *,
        bot_id: str,
        prompt: str,
        label: str,
        latest_gcs_prefix: str,
    ) -> Optional[Dict[str, Any]]:
        query = str(prompt or label).strip()
        if not query or query.lower() in _GENERIC_PROMPT_SEEDS or not latest_gcs_prefix:
            return None
        try:
            import vertexai

            from infrastructure.clients.rag_client import RAG_LOCATION, dedupe_evidence, rerank_evidence, retrieve_for_subquery
            from infrastructure.rag.url_map import resolve_evidence_urls
            from infrastructure.services.indexing_service import ensure_bot_corpus
        except Exception:
            logger.exception("Suggested message prompt pack setup failed bot_id=%s", bot_id)
            return None

        try:
            corpus = ensure_bot_corpus(bot_id)
            if not corpus:
                return None
            vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
        except Exception:
            logger.exception("Suggested message prompt pack corpus init failed bot_id=%s", bot_id)
            return None

        query_variants = [query]
        normalized_label = str(label or "").strip()
        if normalized_label and normalized_label.lower() != query.lower():
            query_variants.append(normalized_label)

        all_evidence: List[Dict[str, str]] = []
        for candidate in query_variants[:2]:
            try:
                all_evidence.extend(retrieve_for_subquery(corpus, candidate, top_k=max(_EVIDENCE_LIMIT * 2, 6)))
            except Exception:
                logger.warning(
                    "Suggested message prompt retrieval failed bot_id=%s query=%s",
                    bot_id,
                    candidate[:120],
                )
        if not all_evidence:
            return None

        evidence = dedupe_evidence(all_evidence)
        bucket_name = (config.GCS_BUCKET or os.environ.get("GCS_BUCKET", "")).strip().split("/", 1)[0]
        if bucket_name:
            resolve_evidence_urls(evidence, bucket_name)
        if len(evidence) > _EVIDENCE_LIMIT and PROJECT_ID:
            try:
                rerank_client = genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)
                evidence = rerank_evidence(rerank_client, query, evidence, top_n=_EVIDENCE_LIMIT)
            except Exception:
                logger.warning("Suggested message prompt rerank failed bot_id=%s", bot_id)
                evidence = evidence[:_EVIDENCE_LIMIT]
        else:
            evidence = evidence[:_EVIDENCE_LIMIT]

        evidence_snippets: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        link_targets: List[Dict[str, Any]] = []
        source_urls: List[str] = []
        seen_links = set()
        for item in evidence:
            snippet = str(item.get("snippet") or "").strip()
            if not snippet:
                continue
            normalized_url = _normalize_http_url(item.get("url"))
            title = str(item.get("title") or (_friendly_label_from_url(normalized_url) if normalized_url else label) or label).strip() or label
            payload = {"url": normalized_url, "snippet": snippet[:_SNIPPET_MAX_CHARS], "title": title}
            evidence_snippets.append(payload)
            citations.append(payload)
            if normalized_url and normalized_url not in seen_links:
                seen_links.add(normalized_url)
                source_urls.append(normalized_url)
                link_targets.append({"url": normalized_url, "label": title, "source_kind": "retrieved_evidence"})
            if len(evidence_snippets) >= _EVIDENCE_LIMIT:
                break
        if not evidence_snippets:
            return None
        return {
            "pack_mode": "retrieval_evidence",
            "source_urls": source_urls,
            "evidence_snippets": evidence_snippets,
            "link_targets": link_targets,
            "instruction": "Answer only from the provided retrieved evidence and links. If the pack does not confirm the answer, say so.",
            "citations": citations,
        }

    def _build_url_payload(
        self,
        *,
        urls: Sequence[str],
        docs_by_key: Dict[str, Dict[str, Any]],
        fallback_label: str,
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_urls = _dedupe_urls(urls)
        if not normalized_urls:
            return None
        evidence_snippets: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        link_targets: List[Dict[str, Any]] = []
        source_urls: List[str] = []
        for url in normalized_urls:
            exact_key, base_key = _url_keys(url)
            doc = docs_by_key.get(json.dumps(exact_key)) if exact_key else None
            if doc is None and base_key:
                doc = docs_by_key.get(json.dumps(base_key))
            label = str((doc or {}).get("title") or _friendly_label_from_url(url) or fallback_label).strip()
            source_urls.append(url)
            link_targets.append({"url": url, "label": label, "source_kind": "source_url"})
            if not doc:
                continue
            snippet = str(doc.get("snippet") or "").strip()
            if not snippet:
                continue
            payload = {"url": url, "snippet": snippet, "title": label}
            evidence_snippets.append(payload)
            citations.append(payload)
            if len(evidence_snippets) >= _EVIDENCE_LIMIT:
                break
        if evidence_snippets:
            return {
                "pack_mode": "page_evidence",
                "source_urls": source_urls,
                "evidence_snippets": evidence_snippets,
                "link_targets": link_targets,
                "instruction": instruction,
                "citations": citations or evidence_snippets,
            }
        fallback_url = normalized_urls[0]
        fallback_label_value = link_targets[0]["label"] if link_targets else fallback_label
        fallback_snippet = f"Relevant page: {fallback_url}"
        return {
            "pack_mode": "action_link",
            "source_urls": source_urls,
            "evidence_snippets": [{"url": fallback_url, "snippet": fallback_snippet, "title": fallback_label_value}],
            "link_targets": link_targets,
            "instruction": instruction,
            "citations": [{"url": fallback_url, "snippet": fallback_snippet, "title": fallback_label_value}],
        }

    def _compute_version_hash(
        self,
        *,
        lang: str,
        item: Dict[str, Any],
        binding: str,
        payload: Dict[str, Any],
        latest_gcs_prefix: str,
    ) -> str:
        raw = json.dumps(
            {
                "lang": _normalize_lang(lang),
                "suggestion": {
                    "id": str(item.get("id") or "").strip(),
                    "label": str(item.get("label") or "").strip(),
                    "prompt": str(item.get("prompt") or item.get("message") or item.get("label") or "").strip(),
                    "type": str(item.get("type") or "").strip(),
                    "urls": _dedupe_urls(item.get("urls") or []),
                },
                "binding": binding,
                "pack_mode": payload.get("pack_mode"),
                "source_urls": payload.get("source_urls") or [],
                "instruction": payload.get("instruction") or "",
                "gcs_prefix": latest_gcs_prefix or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SuggestedMessageFastPathService:
    def __init__(
        self,
        *,
        pack_repo: Optional[PostgresSuggestedMessagePackRepository] = None,
        builder_service: Optional[SuggestedMessagePackBuilderService] = None,
    ) -> None:
        self._pack_repo = pack_repo or PostgresSuggestedMessagePackRepository()
        self._builder = builder_service or SuggestedMessagePackBuilderService(pack_repo=self._pack_repo)

    def try_answer(
        self,
        *,
        bot_id: str,
        widget_config: Dict[str, Any],
        lang: str,
        suggested_message_id: str,
        message: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation_context: Optional[str] = None,
    ) -> SuggestedMessageFastPathResult:
        pack = self._resolve_ready_pack(
            bot_id=bot_id,
            widget_config=widget_config,
            lang=lang,
            suggested_message_id=suggested_message_id,
        )
        if not pack:
            return SuggestedMessageFastPathResult(hit=False, reason="pack_miss")
        try:
            client = self._create_client()
            answer = synthesize_with_evidence(
                client,
                message,
                list(pack.evidence_snippets or pack.citations or []),
                system_instruction=self._build_system_instruction(pack),
                model_name=(model_name or "").strip() or _FAST_PATH_MODEL,
                temperature=temperature if temperature is not None else 0.15,
                conversation_context=conversation_context,
            )
        except Exception as exc:
            logger.warning(
                "Suggested message fast path generation failed bot_id=%s suggested_message_id=%s: %s: %s",
                bot_id,
                suggested_message_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            return SuggestedMessageFastPathResult(
                hit=False,
                reason="generation_error",
                pack=pack,
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
        return SuggestedMessageFastPathResult(
            hit=True,
            reason="pack_hit",
            answer=_ensure_action_link_visible(
                sanitize_answer_citations(answer),
                pack=pack,
            ),
            citations=list(pack.citations or pack.evidence_snippets or []),
            pack=pack,
        )

    def stream_answer(
        self,
        *,
        bot_id: str,
        widget_config: Dict[str, Any],
        lang: str,
        suggested_message_id: str,
        message: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation_context: Optional[str] = None,
    ) -> Tuple[SuggestedMessageFastPathResult, Optional[Iterator[str]]]:
        pack = self._resolve_ready_pack(
            bot_id=bot_id,
            widget_config=widget_config,
            lang=lang,
            suggested_message_id=suggested_message_id,
        )
        if not pack:
            return SuggestedMessageFastPathResult(hit=False, reason="pack_miss"), None
        try:
            client = self._create_client()
        except Exception as exc:
            return (
                SuggestedMessageFastPathResult(
                    hit=False,
                    reason="client_error",
                    pack=pack,
                    error=f"{type(exc).__name__}: {str(exc)[:240]}",
                ),
                None,
            )

        def _generator() -> Iterator[str]:
            for delta in synthesize_with_evidence_stream(
                client,
                message,
                list(pack.evidence_snippets or pack.citations or []),
                system_instruction=self._build_system_instruction(pack),
                model_name=(model_name or "").strip() or _FAST_PATH_MODEL,
                temperature=temperature if temperature is not None else 0.15,
                conversation_context=conversation_context,
            ):
                yield delta

        return (
            SuggestedMessageFastPathResult(
                hit=True,
                reason="pack_hit",
                citations=list(pack.citations or pack.evidence_snippets or []),
                pack=pack,
            ),
            _generator(),
        )

    def _resolve_ready_pack(
        self,
        *,
        bot_id: str,
        widget_config: Dict[str, Any],
        lang: str,
        suggested_message_id: str,
    ) -> Optional[SuggestedMessagePack]:
        # Runtime should not reload crawl artifacts; it uses the latest ready pack
        # produced by rebuild hooks on widget-config changes and post-crawl updates.
        pack = self._pack_repo.get_latest(
            bot_id=bot_id,
            lang=_normalize_lang(lang),
            suggested_message_id=str(suggested_message_id or "").strip(),
        )
        if not pack or str(pack.status or "").strip().lower() != "ready":
            return None
        return pack

    @staticmethod
    def _build_system_instruction(pack: SuggestedMessagePack) -> str:
        base = (
            "This request came from a tapped suggested message. "
            "Answer only from the provided evidence and links. "
            "Keep the reply natural and aligned with the user's wording. "
            "If the pack does not confirm a detail, say that you cannot confirm it from the provided information. "
            "Never invent details, prices, schedules, or policies."
        )
        extra = str(pack.instruction or "").strip()
        return f"{base}\n\n{extra}" if extra else base

    @staticmethod
    def _create_client() -> genai.Client:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID is not configured")
        return genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)


_builder_instance: Optional[SuggestedMessagePackBuilderService] = None
_fast_path_instance: Optional[SuggestedMessageFastPathService] = None


def suggested_message_pack_builder_service() -> SuggestedMessagePackBuilderService:
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = SuggestedMessagePackBuilderService()
    return _builder_instance


def suggested_message_fast_path_service() -> SuggestedMessageFastPathService:
    global _fast_path_instance
    if _fast_path_instance is None:
        _fast_path_instance = SuggestedMessageFastPathService()
    return _fast_path_instance
