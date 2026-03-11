import unittest

from domain.platform_profiles import (
    build_line_design_effective,
    get_line_design_profile,
    get_line_rich_menu_labels_from_suggested_messages,
    normalize_line_design_overrides,
)


class LineDesignProfileTests(unittest.TestCase):
    def test_profile_has_expected_sections(self):
        profile = get_line_design_profile()
        self.assertIn("suggested_actions", profile)
        self.assertIn("asset_carousel", profile)
        self.assertIn("rich_menu", profile)
        self.assertIn("defaults", profile["rich_menu"])
        self.assertIn("actions", profile["rich_menu"]["defaults"])

    def test_normalize_rejects_unknown_action_and_icon(self):
        profile = get_line_design_profile()
        normalized = normalize_line_design_overrides(
            {
                "rich_menu": {
                    "actions": [
                        {
                            "id": "unknown_action",
                            "enabled": True,
                            "icon": "unknown_icon",
                            "labels": {"en": "X", "ja": "X"},
                        }
                    ]
                }
            },
            profile=profile,
        )
        rich = normalized.get("rich_menu") or {}
        self.assertFalse(rich.get("actions"))

    def test_effective_merges_action_label_override(self):
        profile = get_line_design_profile()
        defaults = profile.get("rich_menu", {}).get("defaults", {})
        actions = defaults.get("actions") if isinstance(defaults.get("actions"), list) else []
        if not actions:
            self.skipTest("No editable LINE rich menu actions configured")
        action_id = str(actions[0].get("id") or "").strip()
        if not action_id:
            self.skipTest("First LINE rich menu action has no id")

        effective = build_line_design_effective(
            {
                "rich_menu": {
                    "actions": [
                        {
                            "id": action_id,
                            "labels": {"en": "Custom EN", "ja": "Custom JA"},
                        }
                    ]
                }
            },
            profile=profile,
        )
        rich_actions = effective.get("rich_menu", {}).get("actions") or []
        target = next((item for item in rich_actions if str(item.get("id") or "") == action_id), None)
        self.assertIsNotNone(target)
        self.assertEqual(target.get("labels", {}).get("en"), "Custom EN")
        self.assertEqual(target.get("labels", {}).get("ja"), "Custom JA")

    def test_rich_menu_labels_from_suggested_messages_maps_types(self):
        labels = get_line_rich_menu_labels_from_suggested_messages(
            {
                "language": "en",
                "suggestedMessages": [
                    {"id": "s1", "label": "Book now", "type": "ai_response", "fastPathBinding": "reservation"},
                    {"id": "s2", "label": "See menu", "type": "show_menu"},
                    {"id": "s3", "label": "Talk to staff", "type": "escalate"},
                ],
            },
            lang="en",
        )
        self.assertEqual(labels.get("reserve"), "Book now")
        self.assertEqual(labels.get("menu"), "See menu")
        self.assertEqual(labels.get("support"), "Talk to staff")

    def test_rich_menu_labels_from_suggested_messages_uses_first_match(self):
        labels = get_line_rich_menu_labels_from_suggested_messages(
            {
                "language": "en",
                "suggestedMessages": [
                    {"id": "a", "label": "Menu A", "type": "show_menu"},
                    {"id": "b", "label": "Menu B", "type": "show_menu"},
                ],
            },
            lang="en",
        )
        self.assertEqual(labels.get("menu"), "Menu A")


if __name__ == "__main__":
    unittest.main()
