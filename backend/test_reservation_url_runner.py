import json
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
