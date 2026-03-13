import unittest
from types import SimpleNamespace

from domain.platform_profiles import (
    get_dashboard_create_bot_flow,
    get_default_post_crawl_jobs,
    get_default_source_language,
    get_instagram_support_messages,
    get_job_pipeline_workflow,
    get_knowledge_tabs_for_widget,
    get_line_cancel_keywords,
    get_line_support_messages,
    get_menu_category_aliases,
    get_menu_view_all_url_tokens,
    get_reservation_config_from_widget,
    get_reservation_url_for_platform,
    get_web_support_messages,
    normalize_reservation_links,
)


class ModularConfigTests(unittest.TestCase):
    def test_defaults_are_loaded(self):
        self.assertTrue(get_default_source_language())
        self.assertTrue(get_default_post_crawl_jobs())
        self.assertTrue(get_menu_category_aliases())
        self.assertTrue(get_menu_view_all_url_tokens())

    def test_reservation_links_bridge_supports_legacy_and_map(self):
        widget_config = {
            "reservationLinks": {
                "tabelog": "https://tabelog.com/tokyo/A1304/A130401/13224546/",
            },
            "hotPepperUrl": "www.hotpepper.jp/strJ001234567/",
        }
        normalized = normalize_reservation_links(widget_config)
        self.assertIn("tabelog", normalized)
        self.assertIn("hotpepper", normalized)
        self.assertTrue(normalized["hotpepper"].startswith("https://"))
        self.assertEqual(
            get_reservation_url_for_platform(widget_config, "tabelog"),
            normalized["tabelog"],
        )

    def test_menu_view_link_tokens_are_config_driven(self):
        items = [
            SimpleNamespace(metadata={"source_url": "https://example.com/path"}),
            SimpleNamespace(metadata={"source_url": "https://example.com/party/abc"}),
        ]
        tokens_map = get_menu_view_all_url_tokens()
        urls = [str(i.metadata.get("source_url") or "") for i in items]
        resolved = ""
        for u in urls:
            low = u.lower()
            if any(t.lower() in low for t in tokens_map.get("course", [])):
                resolved = u
                break
        self.assertIn("party", resolved)

    def test_knowledge_tabs_resolve_from_platform_without_url_validation(self):
        widget_config = {"reservationPlatform": "hotpepper"}
        tabs = get_knowledge_tabs_for_widget(widget_config)
        self.assertEqual(["menu"], tabs)

    def test_create_bot_flow_is_loaded_from_config(self):
        flow = get_dashboard_create_bot_flow(lang="en")
        self.assertTrue(flow.get("step_groups"))
        self.assertTrue(flow.get("screen_order"))
        defs = flow.get("screen_definitions") or {}
        self.assertIn("reservation_destination", defs)
        self.assertIn("image_extraction_permission", defs)
        self.assertIn("suggested_messages", defs)
        self.assertEqual("action_destination_url", defs["reservation_destination"].get("component"))
        self.assertEqual("image_extraction_permission", defs["image_extraction_permission"].get("component"))
        self.assertEqual("suggested_messages", defs["suggested_messages"].get("component"))
        order = flow.get("screen_order") or []
        self.assertLess(order.index("image_extraction_permission"), order.index("training"))
        self.assertLess(order.index("suggested_messages"), order.index("widget"))

    def test_reservation_customer_destination_overrides_platform_url(self):
        widget_config = {
            "reservationPlatform": "tabelog",
            "reservationLinks": {
                "tabelog": "https://tabelog.com/tokyo/A1304/A130401/13224546/",
            },
            "actionDestinationLinks": {
                "reservation": "https://example.com/book-now",
            },
        }
        cfg = get_reservation_config_from_widget(widget_config, lang="en")
        self.assertIsNotNone(cfg)
        self.assertEqual("https://example.com/book-now", cfg["url"])
        self.assertEqual("https://tabelog.com/tokyo/A1304/A130401/13224546/", cfg["platform_url"])

    def test_line_support_messages_are_localized(self):
        ja = get_line_support_messages(lang="ja")
        en = get_line_support_messages(lang="en")
        self.assertIn("サポート", ja["prompt"])
        self.assertIn("staff", en["prompt"].lower())
        self.assertIn("キャンセル", ja["cancel_ack"])
        self.assertIn("cancelled", en["cancel_ack"].lower())

    def test_line_cancel_keywords_are_non_empty(self):
        keywords = get_line_cancel_keywords()
        self.assertIn("cancel", keywords)
        self.assertIn("キャンセル", keywords)

    def test_web_support_messages_are_localized(self):
        ja = get_web_support_messages(lang="ja")
        en = get_web_support_messages(lang="en")
        self.assertEqual("サポートに相談", ja["modalTitle"])
        self.assertEqual("Contact support", en["modalTitle"])
        self.assertIn("メールアドレス", ja["modalSubtitle"])
        self.assertIn("email", en["modalSubtitle"].lower())

    def test_instagram_support_messages_are_localized(self):
        ja = get_instagram_support_messages(lang="ja")
        en = get_instagram_support_messages(lang="en")
        self.assertIn("スタッフ", ja["prompt"])
        self.assertIn("staff", en["prompt"].lower())
        self.assertIn("キャンセル", ja["cancel_ack"])
        self.assertIn("cancelled", en["cancel_ack"].lower())


    def test_job_pipeline_workflow_keeps_asset_extraction_by_default(self):
        steps = get_job_pipeline_workflow({}, workflow_id="default")
        self.assertIn("prompt_generation", steps)
        self.assertIn("asset_extraction", steps)
        self.assertNotIn("booking_link", steps)

    def test_job_pipeline_workflow_disables_asset_extraction_when_opted_out(self):
        steps = get_job_pipeline_workflow({"allowAutoImageExtraction": False}, workflow_id="default")
        self.assertNotIn("asset_extraction", steps)
        self.assertIn("prompt_generation", steps)

    def test_job_pipeline_workflow_platform_override_remains_for_menu_platforms(self):
        steps = get_job_pipeline_workflow(
            {"reservationPlatform": "tabelog", "allowAutoImageExtraction": False},
            workflow_id="default",
        )
        self.assertEqual(["prompt_generation", "menu_extraction", "reservation_url"], steps)

    def test_job_pipeline_workflow_tablecheck_override_keeps_booking_link(self):
        steps = get_job_pipeline_workflow(
            {"reservationPlatform": "tablecheck", "allowAutoImageExtraction": False},
            workflow_id="default",
        )
        self.assertEqual(["prompt_generation", "booking_link"], steps)


if __name__ == "__main__":
    unittest.main()
