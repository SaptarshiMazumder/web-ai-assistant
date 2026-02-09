import io
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


PDF_MAX_PAGES = _get_int_env("PDF_MAX_PAGES", 200)
PDF_OCR_MAX_PAGES = _get_int_env("PDF_OCR_MAX_PAGES", 40)
PDF_OCR_IMAGE_SCALE = float(os.environ.get("PDF_OCR_IMAGE_SCALE", "2.0") or "2.0")

PDF_LOG_PREVIEW_CHARS = _get_int_env("PDF_LOG_PREVIEW_CHARS", 1200)
PDF_LOG_MAX_PAGES_PREVIEW = _get_int_env("PDF_LOG_MAX_PAGES_PREVIEW", 8)
PDF_LOG_FULL_CONTENT = (os.environ.get("PDF_LOG_FULL_CONTENT") or "").strip().lower() in ("1", "true", "yes", "y")

PROJECT_ID = (os.environ.get("PROJECT_ID") or "").strip()
GENAI_LOCATION = (os.environ.get("GENAI_LOCATION") or "global").strip()
RAG_LOCATION = (os.environ.get("RAG_LOCATION") or os.environ.get("LOCATION") or "us-central1").strip()
PDF_OCR_MODEL = (os.environ.get("VERTEX_PDF_OCR_MODEL") or os.environ.get("VERTEX_RAG_MODEL") or "gemini-2.0-flash-001").strip()
PDF_ENABLE_GEMINI_OCR = (os.environ.get("PDF_ENABLE_GEMINI_OCR") or "").strip().lower() in ("1", "true", "yes", "y")


def _preview(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if PDF_LOG_FULL_CONTENT:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    t = text.strip()
    if not t:
        return 0.0
    total = max(1, len(t))
    printable = sum(1 for ch in t if ch.isprintable())
    replacement = t.count("\ufffd")
    return (printable / total) - (replacement * 0.02)


def _get_genai_client():
    from google import genai  # type: ignore
    import vertexai  # type: ignore

    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)
    return genai.Client(vertexai=True, project=PROJECT_ID, location=GENAI_LOCATION)


def _gemini_ocr_image(image_bytes: bytes, *, mime_type: str = "image/png") -> str:
    """
    OCR using Gemini Vision. Returns extracted plain text (best-effort).
    Raises on hard failures so caller can fall back.
    """
    client = _get_genai_client()
    from google.genai import types  # type: ignore

    prompt = (
        "You are an OCR engine.\n"
        "Extract ALL readable text from the image.\n"
        "Rules:\n"
        "- Return ONLY the extracted text.\n"
        "- Preserve Japanese and English characters.\n"
        "- Keep line breaks where they appear.\n"
        "- Do not add commentary.\n"
    )
    resp = client.models.generate_content(
        model=PDF_OCR_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    # google-genai supports bytes parts for images
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    return (resp.text or "").strip()


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return bool(shutil.which("tesseract"))


def _tesseract_ocr_image(pil_image) -> str:
    import pytesseract  # type: ignore

    lang = (os.environ.get("PDF_TESSERACT_LANG") or "jpn+eng").strip()
    # pytesseract returns str; preserve newlines
    return (pytesseract.image_to_string(pil_image, lang=lang) or "").strip()


def _render_pdf_pages(pdf_bytes: bytes, *, max_pages: int) -> List[Tuple[int, "object"]]:
    """
    Render pages to PIL Images using pypdfium2. Returns [(page_number_1_indexed, pil_image), ...]
    """
    import pypdfium2 as pdfium  # type: ignore

    images: List[Tuple[int, "object"]] = []
    pdf = pdfium.PdfDocument(pdf_bytes)
    n_pages = len(pdf)
    limit = min(n_pages, max_pages)
    for i in range(limit):
        page = pdf.get_page(i)
        try:
            bitmap = page.render(scale=PDF_OCR_IMAGE_SCALE)
            pil_image = bitmap.to_pil()
            images.append((i + 1, pil_image))
        finally:
            try:
                page.close()
            except Exception:
                pass
    try:
        pdf.close()
    except Exception:
        pass
    return images


@dataclass(frozen=True)
class PdfPageText:
    page: int
    text: str
    method: str  # "pymupdf" | "pdfplumber" | "gemini_ocr" | "tesseract_ocr"


def extract_pdf_pages_text(
    pdf_bytes: bytes,
    *,
    filename: str = "",
    log_prefix: str = "",
) -> List[PdfPageText]:
    """
    Extract text from PDF robustly:
    1) Fast text extraction (default): PyMuPDF (industry standard, very fast)
    2) Fallback: pdfplumber (pdfminer-based) if PyMuPDF fails/empty
    3) Optional OCR: Gemini (if enabled) then Tesseract (if available) for scanned/image-only pages
    """
    if not pdf_bytes:
        return []

    logger.info(
        "%sPDF extract start filename=%s primary=pymupdf fallback=pdfplumber gemini_ocr=%s(model=%s) renderer=pypdfium2 tesseract_fallback=%s max_pages=%d ocr_max_pages=%d",
        log_prefix,
        filename or "",
        "enabled" if PDF_ENABLE_GEMINI_OCR else "disabled",
        PDF_OCR_MODEL,
        "enabled" if _tesseract_available() else "unavailable",
        PDF_MAX_PAGES,
        PDF_OCR_MAX_PAGES,
    )

    extracted: List[PdfPageText] = []
    rendered_images: Optional[dict[int, "object"]] = None

    # PyMuPDF: fastest text-layer extraction (good for most digital PDFs).
    try:
        import fitz  # type: ignore
    except Exception as e:
        fitz = None  # type: ignore
        logger.warning("%sPyMuPDF not available: %s", log_prefix, f"{type(e).__name__}: {str(e)[:120]}")

    # pdfplumber fallback (pdfminer-based)
    import pdfplumber  # type: ignore

    # Determine page count
    total_pages = 0
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = int(doc.page_count)
        except Exception:
            total_pages = 0
        finally:
            try:
                doc.close()  # type: ignore
            except Exception:
                pass
    if total_pages <= 0:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
        except Exception:
            total_pages = 0

    if total_pages > PDF_MAX_PAGES:
        logger.warning("%sPDF %s has %d pages; truncating to %d", log_prefix, filename, total_pages, PDF_MAX_PAGES)
    limit = min(total_pages or 0, PDF_MAX_PAGES) if total_pages else PDF_MAX_PAGES

    # Only render pages if OCR is enabled and needed.
    def ensure_rendered_images() -> dict[int, "object"]:
        nonlocal rendered_images
        if rendered_images is not None:
            return rendered_images
        if not PDF_ENABLE_GEMINI_OCR and not _tesseract_available():
            rendered_images = {}
            return rendered_images
        try:
            render_limit = min(limit, PDF_OCR_MAX_PAGES)
            imgs = _render_pdf_pages(pdf_bytes, max_pages=render_limit)
            rendered_images = {p: img for p, img in imgs}
        except Exception as e:
            logger.warning("%sPDF render failed err=%s", log_prefix, f"{type(e).__name__}: {str(e)[:120]}")
            rendered_images = {}
        return rendered_images

    # Open documents for per-page extraction
    fitz_doc = None
    if fitz is not None:
        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            fitz_doc = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # If total_pages was unknown, use pdfplumber's count.
        if total_pages <= 0:
            total_pages = len(pdf.pages)
            limit = min(total_pages, PDF_MAX_PAGES)

        for idx in range(limit):
            page_num = idx + 1
            final_text = ""
            method = "pymupdf"

            # 1) PyMuPDF (fast)
            if fitz_doc is not None:
                try:
                    p = fitz_doc.load_page(idx)
                    final_text = (p.get_text("text") or "").strip()
                except Exception:
                    final_text = ""

            # 2) pdfplumber fallback
            if not final_text:
                method = "pdfplumber"
                try:
                    final_text = (pdf.pages[idx].extract_text() or "").strip()
                except Exception:
                    final_text = ""

            # 3) Optional OCR if still empty (scanned)
            if not final_text:
                imgs = ensure_rendered_images()
                pil_image = imgs.get(page_num)
                if pil_image is not None and PDF_ENABLE_GEMINI_OCR:
                    try:
                        buf = io.BytesIO()
                        pil_image.save(buf, format="PNG")
                        final_text = (_gemini_ocr_image(buf.getvalue(), mime_type="image/png") or "").strip()
                        if final_text:
                            method = "gemini_ocr"
                    except Exception as e:
                        logger.info("%sGemini OCR failed page=%d err=%s", log_prefix, page_num, f"{type(e).__name__}: {str(e)[:160]}")
                        final_text = ""

                if not final_text and pil_image is not None and _tesseract_available():
                    try:
                        final_text = (_tesseract_ocr_image(pil_image) or "").strip()
                        if final_text:
                            method = "tesseract_ocr"
                    except Exception as e:
                        logger.warning("%sTesseract OCR failed page=%d err=%s", log_prefix, page_num, f"{type(e).__name__}: {str(e)[:160]}")
                        final_text = ""

            extracted.append(PdfPageText(page=page_num, text=final_text, method=method))

            if page_num <= PDF_LOG_MAX_PAGES_PREVIEW and final_text:
                logger.info("%sPDF page %d (%s) preview:\n%s", log_prefix, page_num, method, _preview(final_text, PDF_LOG_PREVIEW_CHARS))

    if fitz_doc is not None:
        try:
            fitz_doc.close()
        except Exception:
            pass

    return extracted
