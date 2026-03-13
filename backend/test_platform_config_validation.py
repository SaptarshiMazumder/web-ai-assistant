import copy
import unittest
from unittest.mock import patch

from domain import platform_profiles
from domain.platform_profiles import (
    get_deterministic_prompt_config,
    get_function_config,
    get_prompt_fallback_config,
    get_prompt_generation_config,
)


class PlatformConfigValidationTests(unittest.TestCase):
    def test_prompt_configs_present(self):
        deterministic = get_deterministic_prompt_config()
        generation = get_prompt_generation_config()
        fallback = get_prompt_fallback_config()

        self.assertIn("personality_with_business_type", deterministic)
        self.assertIn("standard_response_rules_en", generation)
        self.assertIn("personality_en", fallback)

    def test_function_mapping_present(self):
        funcs = get_function_config()
        mapping = funcs.get("suggested_type_to_function")
        self.assertIsInstance(mapping, dict)
        self.assertEqual(mapping.get("show_menu"), "show_menu")

    def test_create_bot_flow_validation_rejects_unknown_component(self):
        cfg = copy.deepcopy(platform_profiles._load_platform_config())
        cfg["dashboard"]["create_bot_flow"]["screen_definitions"]["details"]["component"] = "unknown_component"
        with self.assertRaises(platform_profiles.ConfigValidationError):
            platform_profiles._validate_platform_config(cfg)

    def test_create_bot_flow_validation_rejects_unknown_reservation_platform_visibility(self):
        cfg = copy.deepcopy(platform_profiles._load_platform_config())
        visibility = cfg["dashboard"]["create_bot_flow"]["screen_definitions"]["reservation_destination"].setdefault(
            "visibility", {}
        )
        visibility["reservation_platform_ids"] = ["unknown_platform"]
        with self.assertRaises(platform_profiles.ConfigValidationError):
            platform_profiles._validate_platform_config(cfg)

    def test_create_bot_flow_includes_reservation_platform_visibility(self):
        cfg = copy.deepcopy(platform_profiles._load_platform_config())
        visibility = cfg["dashboard"]["create_bot_flow"]["screen_definitions"]["reservation_destination"].setdefault(
            "visibility", {}
        )
        visibility["reservation_platform_ids"] = ["tabelog", "hotpepper"]
        platform_profiles._validate_platform_config(cfg)
        with patch("domain.platform_profiles._load_platform_config", return_value=cfg):
            flow = platform_profiles.get_dashboard_create_bot_flow(lang="en")
        exported_visibility = flow["screen_definitions"]["reservation_destination"]["visibility"]
        self.assertEqual(
            ["tabelog", "hotpepper"],
            exported_visibility.get("reservation_platform_ids"),
        )


if __name__ == "__main__":
    unittest.main()
