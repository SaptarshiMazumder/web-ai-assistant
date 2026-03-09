import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

BACKEND_DIR = pathlib.Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
sys.modules.setdefault("jwt", types.SimpleNamespace())

from domain.entities import Bot, IndexJob, SuggestedMessagePack
from infrastructure.clients.line_client import build_suggested_flex

_FAKE_DB_PKG = types.ModuleType("infrastructure.db")
_FAKE_DB_REPOS = types.ModuleType("infrastructure.db.repositories")


class _StubRepo:
    def __init__(self, *args, **kwargs) -> None:
        pass


_FAKE_DB_REPOS.PostgresBotAssetRepository = _StubRepo
_FAKE_DB_REPOS.PostgresBotRepository = _StubRepo
_FAKE_DB_REPOS.PostgresIndexJobRepository = _StubRepo
_FAKE_DB_REPOS.PostgresSuggestedMessagePackRepository = _StubRepo
_FAKE_DB_PKG.repositories = _FAKE_DB_REPOS

sys.modules.setdefault("infrastructure.db", _FAKE_DB_PKG)
sys.modules.setdefault("infrastructure.db.repositories", _FAKE_DB_REPOS)

_SPEC = importlib.util.spec_from_file_location(
    "test_suggested_message_pack_service_module",
    BACKEND_DIR / "application" / "services" / "suggested_message_pack_service.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not load suggested_message_pack_service module for tests")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
SuggestedMessagePackBuilderService = _MODULE.SuggestedMessagePackBuilderService
SuggestedMessageFastPathService = _MODULE.SuggestedMessageFastPathService


class _FakeBotRepo:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    def get_bot(self, bot_id: str):
        return self._bot if bot_id == self._bot.bot_id else None


class _FakeIndexRepo:
    def __init__(self, jobs):
        self._jobs = jobs

    def list_jobs_for_bot(self, bot_id: str):
        return list(self._jobs)


class _FakeAssetRepo:
    def __init__(self, assets=None) -> None:
        self._assets = list(assets or [])

    def list_assets_for_bot(self, bot_id: str, *, active_only: bool = False, asset_type: str | None = None):
        return list(self._assets)


class _FakePackRepo:
    def __init__(self) -> None:
        self.replaced = {}

    def replace_for_bot_lang(self, *, bot_id: str, lang: str, packs):
        self.replaced[(bot_id, lang)] = list(packs)


class _FakeReadyPackRepo:
    def __init__(self, pack: SuggestedMessagePack) -> None:
        self._pack = pack

    def get_latest(self, *, bot_id: str, lang: str, suggested_message_id: str):
        if (
            self._pack
            and bot_id == self._pack.bot_id
            and lang == self._pack.lang
            and suggested_message_id == self._pack.suggested_message_id
        ):
            return self._pack
        return None


class SuggestedMessagePackBuilderTests(unittest.TestCase):
    def test_rebuild_creates_page_evidence_pack_for_explicit_urls(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config='{"suggestedMessagesByLanguage":{"en":[{"id":"pricing","label":"Pricing","type":"ai_response","prompt":"Tell me about pricing","urls":["https://example.com/pricing"]}]}}',
        )
        jobs = [
            IndexJob(
                job_id="job_1",
                bot_id="bot_1",
                url="https://example.com",
                hostname="example.com",
                stage="done",
                pages_crawled=1,
                docs_count=1,
                last_crawled_url="https://example.com/pricing",
                last_depth=0,
                gcs_prefix="saas/org/bot/example/20260309",
                last_error="",
                created_at="2026-03-09T00:00:00+00:00",
                updated_at="2026-03-09T00:00:00+00:00",
            )
        ]
        pack_repo = _FakePackRepo()
        service = SuggestedMessagePackBuilderService(
            bot_repo=_FakeBotRepo(bot),
            index_repo=_FakeIndexRepo(jobs),
            asset_repo=_FakeAssetRepo(),
            pack_repo=pack_repo,
        )

        with patch(
            f"{_SPEC.name}._load_docs_from_gcs_prefix",
            return_value=[
                {
                    "url": "https://example.com/pricing",
                    "content": "Source URL: https://example.com/pricing\n\n# Pricing\n\nPlans start at $10 per month.\n\nAnnual plans are discounted.",
                }
            ],
        ):
            result = service.rebuild_for_bot("bot_1")

        self.assertEqual({"langs": 2, "packs": 1}, result)
        packs = pack_repo.replaced[("bot_1", "en")]
        self.assertEqual(1, len(packs))
        self.assertEqual("page_evidence", packs[0].pack_mode)
        self.assertEqual(["https://example.com/pricing"], packs[0].source_urls)
        self.assertIn("Plans start at $10 per month.", packs[0].citations[0]["snippet"])
        self.assertEqual([], pack_repo.replaced[("bot_1", "ja")])

    def test_rebuild_skips_unmapped_ai_response_suggestions(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config='{"suggestedMessagesByLanguage":{"en":[{"id":"ask","label":"Ask a question","type":"ai_response","prompt":"Ask a question"}]}}',
        )
        pack_repo = _FakePackRepo()
        service = SuggestedMessagePackBuilderService(
            bot_repo=_FakeBotRepo(bot),
            index_repo=_FakeIndexRepo([]),
            asset_repo=_FakeAssetRepo(),
            pack_repo=pack_repo,
        )

        result = service.rebuild_for_bot("bot_1")

        self.assertEqual({"langs": 2, "packs": 0}, result)
        self.assertEqual([], pack_repo.replaced[("bot_1", "en")])
        self.assertEqual([], pack_repo.replaced[("bot_1", "ja")])

    def test_rebuild_creates_retrieval_pack_for_plain_ai_response_suggestion(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config='{"suggestedMessagesByLanguage":{"en":[{"id":"pricing","label":"Pricing","type":"ai_response","prompt":"Tell me about pricing"}]}}',
        )
        jobs = [
            IndexJob(
                job_id="job_1",
                bot_id="bot_1",
                url="https://example.com",
                hostname="example.com",
                stage="done",
                pages_crawled=1,
                docs_count=1,
                last_crawled_url="https://example.com/pricing",
                last_depth=0,
                gcs_prefix="saas/org/bot/example/20260309",
                last_error="",
                created_at="2026-03-09T00:00:00+00:00",
                updated_at="2026-03-09T00:00:00+00:00",
            )
        ]
        pack_repo = _FakePackRepo()
        service = SuggestedMessagePackBuilderService(
            bot_repo=_FakeBotRepo(bot),
            index_repo=_FakeIndexRepo(jobs),
            asset_repo=_FakeAssetRepo(),
            pack_repo=pack_repo,
        )

        with patch.object(
            service,
            "_build_prompt_payload",
            return_value={
                "pack_mode": "retrieval_evidence",
                "source_urls": ["https://example.com/pricing"],
                "evidence_snippets": [
                    {
                        "url": "https://example.com/pricing",
                        "title": "Pricing",
                        "snippet": "Plans start at $10 per month.",
                    }
                ],
                "link_targets": [
                    {
                        "url": "https://example.com/pricing",
                        "label": "Pricing",
                        "source_kind": "retrieved_evidence",
                    }
                ],
                "instruction": "Answer only from the provided retrieved evidence and links.",
                "citations": [
                    {
                        "url": "https://example.com/pricing",
                        "title": "Pricing",
                        "snippet": "Plans start at $10 per month.",
                    }
                ],
            },
        ):
            result = service.rebuild_for_bot("bot_1")

        self.assertEqual(2, result["langs"])
        self.assertGreaterEqual(result["packs"], 1)
        pack = pack_repo.replaced[("bot_1", "en")][0]
        self.assertEqual("retrieval_evidence", pack.pack_mode)
        self.assertEqual("pricing", pack.suggested_message_id)
        self.assertEqual(["https://example.com/pricing"], pack.source_urls)
        self.assertIn("Plans start at $10 per month.", pack.citations[0]["snippet"])

    def test_rebuild_creates_reservation_action_link_pack(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=(
                '{"reservationPlatform":"hotpepper","hotPepperUrl":"https://www.hotpepper.jp/strJ001234567/",'
                '"suggestedMessagesByLanguage":{"en":[{"id":"reserve","label":"Book now","type":"ai_response","prompt":"How do I book?","fastPathBinding":"reservation"}]}}'
            ),
        )
        pack_repo = _FakePackRepo()
        service = SuggestedMessagePackBuilderService(
            bot_repo=_FakeBotRepo(bot),
            index_repo=_FakeIndexRepo([]),
            asset_repo=_FakeAssetRepo(),
            pack_repo=pack_repo,
        )

        result = service.rebuild_for_bot("bot_1")

        self.assertEqual(2, result["langs"])
        self.assertGreaterEqual(result["packs"], 1)
        pack = pack_repo.replaced[("bot_1", "en")][0]
        self.assertEqual("action_link", pack.pack_mode)
        self.assertEqual("reserve", pack.suggested_message_id)
        self.assertTrue(pack.source_urls[0].startswith("https://www.hotpepper.jp/strJ001234567/"))
        self.assertTrue(any(target.get("source_kind") == "reservation" for target in pack.link_targets))


class SuggestedMessageFastPathTests(unittest.TestCase):
    def test_action_link_answer_appends_url_when_model_omits_it(self) -> None:
        pack = SuggestedMessagePack(
            pack_id="pack_1",
            bot_id="bot_1",
            org_id="org_1",
            lang="ja",
            suggested_message_id="reserve",
            label="予約",
            prompt="予約したい",
            pack_mode="action_link",
            status="ready",
            version_hash="v1",
            source_urls=["https://www.hotpepper.jp/strJ001234567/"],
            evidence_snippets=[
                {
                    "url": "https://www.hotpepper.jp/strJ001234567/",
                    "title": "予約",
                    "snippet": "Official reservation link: https://www.hotpepper.jp/strJ001234567/",
                }
            ],
            link_targets=[
                {
                    "url": "https://www.hotpepper.jp/strJ001234567/",
                    "label": "予約ページ",
                    "source_kind": "reservation",
                }
            ],
            instruction="Use this reservation link only.",
            citations=[
                {
                    "url": "https://www.hotpepper.jp/strJ001234567/",
                    "title": "予約",
                    "snippet": "Official reservation link: https://www.hotpepper.jp/strJ001234567/",
                }
            ],
        )
        service = SuggestedMessageFastPathService(
            pack_repo=_FakeReadyPackRepo(pack),
            builder_service=object(),
        )

        with patch.object(service, "_create_client", return_value=object()):
            with patch(
                f"{_SPEC.name}.synthesize_with_evidence",
                return_value="はい、ご予約ですね。こちらのページからご予約を承っております。",
            ):
                result = service.try_answer(
                    bot_id="bot_1",
                    widget_config={},
                    lang="ja",
                    suggested_message_id="reserve",
                    message="予約したい",
                )

        self.assertTrue(result.hit)
        self.assertIn("https://www.hotpepper.jp/strJ001234567/", result.answer)


class LineSuggestedFlexTests(unittest.TestCase):
    def test_build_suggested_flex_uses_postback_id(self) -> None:
        flex = build_suggested_flex(
            [
                {"id": "suggest_1", "label": "Pricing", "type": "ai_response"},
            ]
        )
        self.assertIsNotNone(flex)
        action = flex["contents"]["body"]["contents"][0]["action"]
        self.assertEqual("postback", action["type"])
        self.assertEqual("Pricing", action["displayText"])
        self.assertEqual("lineux:suggest:suggest_1", action["data"])


if __name__ == "__main__":
    unittest.main()
