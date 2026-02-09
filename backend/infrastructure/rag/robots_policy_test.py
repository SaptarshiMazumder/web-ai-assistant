import unittest
from urllib.robotparser import RobotFileParser


class RobotsPolicyParseTests(unittest.TestCase):
    def test_disallow_specific_path(self):
        text = "\n".join(
            [
                "User-agent: WebAIbot",
                "Disallow: /private",
                "Allow: /",
            ]
        )
        rp = RobotFileParser()
        rp.parse(text.splitlines())
        self.assertTrue(rp.can_fetch("WebAIbot", "https://example.com/"))
        self.assertTrue(rp.can_fetch("WebAIbot", "https://example.com/public"))
        self.assertFalse(rp.can_fetch("WebAIbot", "https://example.com/private"))
        self.assertFalse(rp.can_fetch("WebAIbot", "https://example.com/private/page"))

    def test_fallback_star_group(self):
        text = "\n".join(
            [
                "User-agent: *",
                "Disallow: /nope",
            ]
        )
        rp = RobotFileParser()
        rp.parse(text.splitlines())
        self.assertFalse(rp.can_fetch("WebAIbot", "https://example.com/nope"))
        self.assertTrue(rp.can_fetch("WebAIbot", "https://example.com/ok"))


if __name__ == "__main__":
    unittest.main()

