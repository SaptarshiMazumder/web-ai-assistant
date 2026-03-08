import importlib.util
import pathlib
import sys
import unittest

BACKEND_DIR = pathlib.Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "test_answer_normalization_service_module",
    BACKEND_DIR / "application" / "services" / "answer_normalization_service.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not load answer_normalization_service module for tests")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RENDER_TARGET_PLAIN_TEXT_CHANNEL = _MODULE.RENDER_TARGET_PLAIN_TEXT_CHANNEL
RENDER_TARGET_WEB_MARKDOWN = _MODULE.RENDER_TARGET_WEB_MARKDOWN
build_answer_link_candidates = _MODULE.build_answer_link_candidates
normalize_answer_links = _MODULE.normalize_answer_links


class AnswerNormalizationServiceTests(unittest.TestCase):
    def test_recovers_japanese_suffix_and_canonicalizes_reservation_url(self) -> None:
        candidates = build_answer_link_candidates(
            reservation_config={
                "url": "https://tabelog.com/tokyo/A1304/A130401/13224546/",
                "link_label": "Reserve on Tabelog",
                "domain_key": "tabelog.com",
            }
        )
        text = "予約はこちら https://tabelog.com/tokyo/A1304/A130401/99999999/から予約できます。"

        web_answer = normalize_answer_links(
            text,
            candidates=candidates,
            render_target=RENDER_TARGET_WEB_MARKDOWN,
        )
        plain_answer = normalize_answer_links(
            text,
            candidates=candidates,
            render_target=RENDER_TARGET_PLAIN_TEXT_CHANNEL,
        )

        self.assertEqual(
            "予約はこちら [Reserve on Tabelog](https://tabelog.com/tokyo/A1304/A130401/13224546/)から予約できます。",
            web_answer.text,
        )
        self.assertEqual(
            "予約はこちら Reserve on Tabelog: https://tabelog.com/tokyo/A1304/A130401/13224546/ から予約できます。",
            plain_answer.text,
        )
        self.assertEqual(
            ["https://tabelog.com/tokyo/A1304/A130401/13224546/"],
            [link.url for link in web_answer.links],
        )

    def test_normalizes_multiple_urls_in_one_answer(self) -> None:
        candidates = build_answer_link_candidates(
            url_bank=[{"label": "Pricing", "url": "https://example.com/pricing"}],
            sources=[{"url": "https://example.com/contact", "excerpt": "Contact us"}],
        )
        text = "See https://example.com/pricing and https://example.com/contact."

        normalized = normalize_answer_links(
            text,
            candidates=candidates,
            render_target=RENDER_TARGET_WEB_MARKDOWN,
        )

        self.assertEqual(
            "See [Pricing](https://example.com/pricing) and [contact page](https://example.com/contact).",
            normalized.text,
        )
        self.assertEqual(2, len(normalized.links))
        self.assertEqual("Pricing", normalized.links[0].label)
        self.assertEqual("contact page", normalized.links[1].label)

    def test_handles_markdown_links_queries_fragments_and_parentheses(self) -> None:
        text = "Read [source](https://example.com/docs/foo(bar)?q=1#top)."

        normalized = normalize_answer_links(
            text,
            candidates=[],
            render_target=RENDER_TARGET_WEB_MARKDOWN,
        )

        self.assertEqual(
            "Read [foo(bar) page](https://example.com/docs/foo(bar)?q=1#top).",
            normalized.text,
        )
        self.assertEqual("https://example.com/docs/foo(bar)?q=1#top", normalized.links[0].url)

    def test_does_not_mutate_plain_text_without_urls(self) -> None:
        text = "営業時間は毎日11:00から22:00です。"

        normalized = normalize_answer_links(
            text,
            candidates=[],
            render_target=RENDER_TARGET_WEB_MARKDOWN,
        )

        self.assertEqual(text, normalized.text)
        self.assertEqual([], normalized.links)

    def test_rewrites_reservation_domain_to_canonical_url(self) -> None:
        candidates = build_answer_link_candidates(
            reservation_config={
                "url": "https://www.hotpepper.jp/strJ001234567/",
                "link_label": "Reserve on Hot Pepper",
                "domain_key": "hotpepper.jp",
            }
        )
        text = "Book here: https://www.hotpepper.jp/strJ009999999/?vos=foo"

        normalized = normalize_answer_links(
            text,
            candidates=candidates,
            render_target=RENDER_TARGET_WEB_MARKDOWN,
        )

        self.assertEqual(
            "Book here: [Reserve on Hot Pepper](https://www.hotpepper.jp/strJ001234567/)",
            normalized.text,
        )


if __name__ == "__main__":
    unittest.main()
