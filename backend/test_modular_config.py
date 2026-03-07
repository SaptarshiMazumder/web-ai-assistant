import unittest
from types import SimpleNamespace

from domain.platform_profiles import (
    get_default_post_crawl_jobs,
    get_default_source_language,
    get_menu_category_aliases,
    get_menu_view_all_url_tokens,
    get_reservation_url_for_platform,
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


if __name__ == "__main__":
    unittest.main()
