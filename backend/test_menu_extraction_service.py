import importlib.util
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

from application.services.menu_extraction_service import (
    _extract_tablecheck_menu_items,
    _is_profile_menu_candidate_url,
)
from domain.platform_profiles import resolve_platform_profile


_HAS_BS4 = importlib.util.find_spec("bs4") is not None


@unittest.skipUnless(_HAS_BS4, "BeautifulSoup is required for deterministic menu extractor tests")
class TableCheckMenuExtractionTests(unittest.TestCase):
    def test_tablecheck_extractor_parses_name_price_details_and_image(self) -> None:
        html = """
        <html><body>
          <div class="menu-item show-more-expander">
            <div class="menu-item-data" data-name="Seafood Paella"></div>
            <div class="menu-item-tagline">Fresh seafood</div>
            <div class="menu-item-small">¥ 3,300 (Tax incl.)</div>
            <div class="menu-item-text">Chef special Read more</div>
            <img src="https://cdn2.tablecheck.com/menu_items/abc/images/md/item.jpg" />
          </div>
          <div class="menu-item show-more-expander">
            <div class="menu-item-data" data-name=""></div>
            <div class="menu-item-name">Steak Course</div>
            <div class="menu-item-text">Great dinner 4,500円 Fine Print details Valid Dates 2026-01-01</div>
            <a data-lightbox="menu-item-2" href="https://cdn3.tablecheck.com/menu_items/def/images/xl/item.jpg"></a>
          </div>
          <div class="menu-item show-more-expander">
            <div class="menu-item-data" data-name="Course Menu"></div>
          </div>
        </body></html>
        """
        with patch(
            "application.services.menu_extraction_service._fetch_html",
            return_value=html,
        ):
            items = _extract_tablecheck_menu_items("https://www.tablecheck.com/en/shops/sample/reserve")

        self.assertEqual(2, len(items))
        names = {str(item.get("name") or "") for item in items}
        self.assertIn("Seafood Paella", names)
        self.assertIn("Steak Course", names)
        self.assertNotIn("Course Menu", names)

        seafood = next(item for item in items if item.get("name") == "Seafood Paella")
        steak = next(item for item in items if item.get("name") == "Steak Course")

        self.assertIn("3,300", str(seafood.get("price_text") or ""))
        self.assertEqual(
            "https://cdn2.tablecheck.com/menu_items/abc/images/md/item.jpg",
            seafood.get("image_url"),
        )
        self.assertIn("Chef special", str(seafood.get("details") or ""))

        self.assertIn("4,500", str(steak.get("price_text") or ""))
        self.assertEqual(
            "https://cdn3.tablecheck.com/menu_items/def/images/xl/item.jpg",
            steak.get("image_url"),
        )
        self.assertNotIn("Fine Print", str(steak.get("details") or ""))

    def test_tablecheck_extractor_returns_empty_when_no_menu_blocks(self) -> None:
        with patch(
            "application.services.menu_extraction_service._fetch_html",
            return_value="<html><body><h1>No menu</h1></body></html>",
        ):
            items = _extract_tablecheck_menu_items("https://www.tablecheck.com/en/shops/sample/reserve")

        self.assertEqual([], items)

    def test_tablecheck_extractor_canonicalizes_shop_url_to_reserve(self) -> None:
        html = """
        <html><body>
          <div class="menu-item show-more-expander">
            <div class="menu-item-data" data-name="Lunch Set"></div>
            <div class="menu-item-small">¥ 2,500</div>
          </div>
        </body></html>
        """
        with patch(
            "application.services.menu_extraction_service._fetch_html",
            return_value=html,
        ) as fetch_mock:
            items = _extract_tablecheck_menu_items("https://www.tablecheck.com/en/shops/sample")

        fetch_mock.assert_called_once_with("https://www.tablecheck.com/en/shops/sample/reserve/")
        self.assertEqual(1, len(items))
        self.assertEqual("https://www.tablecheck.com/en/shops/sample/reserve/", items[0].get("link_url"))

    def test_tablecheck_shop_url_is_menu_candidate(self) -> None:
        profile, _ = resolve_platform_profile("https://www.tablecheck.com/en/shops/sample")
        self.assertIsNotNone(profile)
        self.assertTrue(_is_profile_menu_candidate_url("https://www.tablecheck.com/en/shops/sample", profile))


if __name__ == "__main__":
    unittest.main()
