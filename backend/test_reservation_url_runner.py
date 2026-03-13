import json
import sys
import types
import unittest
from unittest.mock import patch

if "jwt" not in sys.modules:
    sys.modules["jwt"] = types.ModuleType("jwt")

if "psycopg" not in sys.modules:
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda *args, **kwargs: None  # pragma: no cover

    class _PsycopgErrors(types.ModuleType):
        def __getattr__(self, name: str):  # pragma: no cover
            return Exception

    fake_errors = _PsycopgErrors("psycopg.errors")
    fake_errors.OperationalError = Exception
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.tuple_row = object()

    fake_psycopg.errors = fake_errors
    fake_psycopg.rows = fake_rows
    sys.modules["psycopg"] = fake_psycopg
    sys.modules["psycopg.errors"] = fake_errors
    sys.modules["psycopg.rows"] = fake_rows

from application.services.job_pipeline.runners.reservation_url import (
    ReservationUrlRunner,
    build_base_reservation_url,
    pick_reservation_url_candidate,
)
from domain.entities import Bot


class _FakeBotRepo:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot
        self.updated: str | None = None

    def get_bot(self, bot_id: str) -> Bot | None:
        return self._bot if bot_id == self._bot.bot_id else None

    def update_widget_config(self, bot_id: str, config_json: str) -> None:
        if bot_id == self._bot.bot_id:
            self.updated = config_json
            self._bot.widget_config = config_json


class ReservationUrlRunnerTests(unittest.TestCase):
    def test_build_base_reservation_url_uses_pattern(self) -> None:
        base = build_base_reservation_url(
            "https://www.hotpepper.jp/strJ001234567/coupon/?foo=1",
            {"base_path_pattern": "^/(?:[a-z]{2}/)?strJ\\d+/?"},
        )
        self.assertEqual("https://www.hotpepper.jp/strJ001234567/", base)

    def test_build_base_reservation_url_tablecheck_short_path_adds_reserve(self) -> None:
        base = build_base_reservation_url(
            "https://www.tablecheck.com/ja/yurakuchokakida",
            {
                "base_path_pattern": "^/(?:[a-z]{2}(?:-[A-Za-z]{2})?/)?(?:shops/)?[A-Za-z0-9._-]+/reserve(?:/|$)",
            },
        )
        self.assertEqual("https://www.tablecheck.com/ja/yurakuchokakida/reserve/", base)

    def test_candidate_selection_prefers_matching_domain_and_path(self) -> None:
        rule = {
            "allowed_domains": ["hotpepper.jp"],
            "include_path_patterns": ["^/(?:[a-z]{2}/)?strJ\\d+/?$"],
            "exclude_path_patterns": ["/map", "/coupon"],
            "path_keyword_scores": {"strj": 8, "reserve": 6},
        }
        candidate = pick_reservation_url_candidate(
            [
                "https://www.hotpepper.jp/map",
                "https://www.hotpepper.jp/strJ001234567/",
                "https://example.com/strJ001234567/",
            ],
            rule,
        )
        self.assertEqual("https://www.hotpepper.jp/strJ001234567/", candidate)

    def test_runner_does_not_overwrite_manual_url(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "tabelog",
                    "tabelogUrl": "https://tabelog.com/manual/keep/",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with patch(
            "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
            return_value=fake_repo,
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "crawled_urls": ["https://tabelog.com/tokyo/A1304/A130401/13224546/"],
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("url_already_present", result.output.get("reason"))
        self.assertIsNone(fake_repo.updated)

    def test_runner_autofills_when_missing(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "hotpepper",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with patch(
            "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
            return_value=fake_repo,
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://www.hotpepper.jp/strJ001234567/",
                    "crawled_urls": [
                        "https://www.hotpepper.jp/map",
                        "https://www.hotpepper.jp/strJ001234567/",
                    ],
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("updated", result.output.get("status"))
        self.assertEqual("base_url", result.output.get("assignment_mode"))
        self.assertIsNotNone(fake_repo.updated)
        cfg = json.loads(fake_repo.updated or "{}")
        self.assertEqual("https://www.hotpepper.jp/strJ001234567/", cfg.get("hotPepperUrl"))
        self.assertEqual(
            "https://www.hotpepper.jp/strJ001234567/",
            (cfg.get("reservationLinks") or {}).get("hotpepper"),
        )

    def test_runner_can_use_candidate_mode_when_configured(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "hotpepper",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with (
            patch(
                "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
                return_value=fake_repo,
            ),
            patch(
                "application.services.job_pipeline.runners.reservation_url.get_reservation_url_rule",
                return_value={
                    "assignment_mode": "candidate",
                    "allowed_domains": ["hotpepper.jp"],
                    "include_path_patterns": ["^/(?:[a-z]{2}/)?strJ\\d+/?$"],
                    "exclude_path_patterns": ["/map"],
                    "path_keyword_scores": {"strj": 8},
                },
            ),
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://www.hotpepper.jp/map",
                    "crawled_urls": [
                        "https://www.hotpepper.jp/strJ001999999/",
                    ],
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("updated", result.output.get("status"))
        self.assertEqual("candidate", result.output.get("assignment_mode"))
        cfg = json.loads(fake_repo.updated or "{}")
        self.assertEqual("https://www.hotpepper.jp/strJ001999999/", cfg.get("hotPepperUrl"))

    def test_runner_tablecheck_base_url_canonicalizes_when_valid(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "tablecheck",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with patch(
            "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
            return_value=fake_repo,
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://www.tablecheck.com/en/shops/craftcircus/reserve?foo=1",
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("updated", result.output.get("status"))
        self.assertEqual("base_url", result.output.get("assignment_mode"))
        cfg = json.loads(fake_repo.updated or "{}")
        self.assertEqual("https://www.tablecheck.com/en/shops/craftcircus/reserve/", cfg.get("tableCheckUrl"))
        self.assertEqual(
            "https://www.tablecheck.com/en/shops/craftcircus/reserve/",
            (cfg.get("reservationLinks") or {}).get("tablecheck"),
        )

    def test_runner_tablecheck_short_root_url_canonicalizes_to_reserve(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "tablecheck",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with patch(
            "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
            return_value=fake_repo,
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://www.tablecheck.com/ja/yurakuchokakida",
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("updated", result.output.get("status"))
        cfg = json.loads(fake_repo.updated or "{}")
        self.assertEqual("https://www.tablecheck.com/ja/yurakuchokakida/reserve/", cfg.get("tableCheckUrl"))

    def test_runner_base_url_can_use_crawled_fallback_by_default(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "hotpepper",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with (
            patch(
                "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
                return_value=fake_repo,
            ),
            patch(
                "application.services.job_pipeline.runners.reservation_url.get_reservation_url_rule",
                return_value={
                    "assignment_mode": "base_url",
                    "base_url_source_key": "root_url",
                    "base_path_pattern": "^/(?:[a-z]{2}/)?strJ\\d+/?",
                    "allowed_domains": ["hotpepper.jp"],
                    "include_path_patterns": ["^/(?:[a-z]{2}/)?strJ\\d+/?$"],
                    "exclude_path_patterns": ["/map"],
                },
            ),
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://example.com/not-hotpepper",
                    "crawled_urls": [
                        "https://www.hotpepper.jp/map",
                        "https://www.hotpepper.jp/strJ001999999/",
                    ],
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("updated", result.output.get("status"))
        cfg = json.loads(fake_repo.updated or "{}")
        self.assertEqual("https://www.hotpepper.jp/strJ001999999/", cfg.get("hotPepperUrl"))

    def test_runner_base_url_skips_when_crawled_fallback_disabled(self) -> None:
        bot = Bot(
            bot_id="bot_1",
            org_id="org_1",
            display_name="Bot",
            publishable_key="pk",
            secret_key="sk",
            widget_config=json.dumps(
                {
                    "reservationPlatform": "tablecheck",
                }
            ),
        )
        fake_repo = _FakeBotRepo(bot)
        runner = ReservationUrlRunner()

        with (
            patch(
                "application.services.job_pipeline.runners.reservation_url.PostgresBotRepository",
                return_value=fake_repo,
            ),
            patch(
                "application.services.job_pipeline.runners.reservation_url.get_reservation_url_rule",
                return_value={
                    "assignment_mode": "base_url",
                    "base_url_source_key": "root_url",
                    "base_path_pattern": "^/(?:[a-z]{2}(?:-[A-Za-z]{2})?/)?(?:shops/)?[A-Za-z0-9._-]+/reserve(?:/|$)",
                    "allowed_domains": ["tablecheck.com"],
                    "allow_crawled_fallback": False,
                },
            ),
        ):
            result = runner.run(
                {
                    "bot_id": "bot_1",
                    "root_url": "https://example.com/restaurant",
                    "crawled_urls": [
                        "https://www.tablecheck.com/en/shops/craftcircus/reserve",
                    ],
                }
            )

        self.assertEqual("done", result.status)
        self.assertEqual("no_base_url_found", result.output.get("reason"))
        self.assertIsNone(fake_repo.updated)


if __name__ == "__main__":
    unittest.main()
