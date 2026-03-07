import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent


class ArchitectureGuardTests(unittest.TestCase):
    def _read(self, rel_path: str) -> str:
        return (ROOT / rel_path).read_text(encoding="utf-8")

    def test_saas_has_no_platform_url_if_else_branch(self):
        text = self._read("api/routes/saas.py")
        self.assertNotIn('if platform_name == "hotpepper"', text)
        self.assertNotIn('elif platform_name == "tabelog"', text)

    def test_line_menu_tokens_are_not_hardcoded_in_route(self):
        text = self._read("api/routes/line_webhook.py")
        self.assertNotIn('"/party"', text)
        self.assertNotIn('"/dtlmenu"', text)

    def test_crawl_source_language_is_not_hardcoded(self):
        text = self._read("infrastructure/tasks/crawl_tasks.py")
        self.assertNotIn("Hardcode JP for now", text)
        self.assertIn("get_default_source_language()", text)


if __name__ == "__main__":
    unittest.main()
