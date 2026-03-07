import unittest

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


if __name__ == "__main__":
    unittest.main()
