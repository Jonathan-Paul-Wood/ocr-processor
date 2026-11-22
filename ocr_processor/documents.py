"""PDFium and Tesseract run only inside the disposable worker process."""
import math
from contextlib import closing
import os
from pathlib import Path
import subprocess
import time
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import pypdfium2 as pdfium

from .errors import ServiceError, error_context


def recognize(image, directory, settings, deadline):
    image_path = Path(directory) / "ocr.png"
    output = Path(directory) / "ocr-result"
    image.save(image_path)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ServiceError("timeout", "Document processing exceeded its time limit.", 504,
                           details={"stage": "ocr", "timeout_seconds": settings.timeout})
    timeout = min(settings.ocr_timeout, remaining)
    try:
        with (Path(directory) / "ocr-stderr.log").open("wb") as diagnostics:
            subprocess.run(
                [settings.tesseract_cmd, str(image_path), str(output), "-l", settings.ocr_language, "--psm", "3"],
                check=True, stdout=subprocess.DEVNULL, stderr=diagnostics,
                timeout=timeout,
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
            )
    except FileNotFoundError:
        raise ServiceError("ocr_unavailable", "Tesseract is not installed or configured.", 503,
                           details={"stage": "ocr", "reason": "executable_not_found",
                                    "hint": "Install Tesseract and ensure TESSERACT_CMD identifies an executable available to the service user."}) from None
    except subprocess.TimeoutExpired:
        raise ServiceError("timeout", "OCR exceeded its time limit.", 504,
                           details={"stage": "ocr", "timeout_seconds": round(timeout, 3),
                                    "ocr_timeout_seconds": settings.ocr_timeout}) from None
    except subprocess.CalledProcessError as exc:
        with (Path(directory) / "ocr-stderr.log").open("rb") as diagnostics:
            stderr = diagnostics.read(8192).decode("utf-8", errors="replace").lower()
        reason, hint = "tesseract_nonzero_exit", "Check installed OCR language packs, image validity, and the container's available memory."
        if any(marker in stderr for marker in ("error opening data file", "failed loading language", "couldn't load any languages")):
            reason, hint = "language_data_unavailable", "Install the trained-data packages for OCR_LANGUAGE and check TESSDATA_PREFIX and file permissions."
        elif "permission denied" in stderr:
            reason, hint = "permission_denied", "Check that the service user can read OCR inputs and write temporary output."
        elif "pixread" in stderr or "cannot read input file" in stderr:
            reason, hint = "image_read_failed", "Tesseract could not decode its rendered input image."
        raise ServiceError("ocr_failed", "OCR failed; check the document and installed language packs.",
                           details={"stage": "ocr", "exit_code": exc.returncode, "reason": reason,
                                    "language": settings.ocr_language, "hint": hint}) from None
    with error_context(stage="ocr_output", operation="read_text_output"):
        return output.with_suffix(".txt").read_text(encoding="utf-8").strip()


def read_documents(files, directory, settings, deadline, force_ocr=False):
    pages = []
    chars = 0
    warnings_list = []

    def append(document, number, text, method):
        nonlocal chars
        if len(pages) >= settings.max_pages:
            raise ServiceError("page_limit", "Combined page count exceeds the limit.", 413,
                               details={"limit": "pages", "actual": len(pages) + 1, "maximum": settings.max_pages})
        chars += len(text)
        if chars > settings.max_text_chars:
            raise ServiceError("text_limit", "Extracted text exceeds the limit; split the documents.", 413,
                               details={"limit": "text_characters", "actual": chars, "maximum": settings.max_text_chars})
        pages.append({"document": document, "page": number, "method": method, "text": text})
        if not text.strip():
            warnings_list.append(f"Document {document}, page {number} contains no readable text.")

    def read_file(index, filename):
        path = Path(filename)
        with path.open("rb") as source:
            signature = source.read(5)
        if signature == b"%PDF-":
            try:
                with pdfium.PdfDocument(path) as pdf:
                    if len(pdf) == 0:
                        raise ServiceError("invalid_document", "PDF has no pages.", details={"reason": "empty_pdf", "pages": 0})
                    if len(pdf) + len(pages) > settings.max_pages:
                        raise ServiceError("page_limit", "Combined page count exceeds the limit.", 413,
                                           details={"limit": "pages", "actual": len(pdf) + len(pages), "maximum": settings.max_pages,
                                                    "document_pages": len(pdf), "previous_pages": len(pages)})
                    for number in range(len(pdf)):
                        with error_context(stage="pdf_page", page=number + 1), closing(pdf[number]) as page:
                            with closing(page.get_textpage()) as textpage:
                                if textpage.count_chars() > settings.max_text_chars:
                                    raise ServiceError("text_limit", "Page text exceeds the limit.", 413,
                                                       details={"limit": "text_characters", "actual": textpage.count_chars(),
                                                                "maximum": settings.max_text_chars, "scope": "page"})
                                text = textpage.get_text_bounded().strip()
                            method = "embedded"
                            if force_ocr or len(text) < 40:
                                width, height = page.get_size()
                                scale = 250 / 72
                                if not all(math.isfinite(x) and x > 0 for x in (width, height)):
                                    raise ServiceError("invalid_document", "PDF page has invalid dimensions.", details={"reason": "invalid_page_dimensions"})
                                render_width, render_height = math.ceil(width * scale), math.ceil(height * scale)
                                if render_width * render_height > settings.max_pixels:
                                    raise ServiceError("pixel_limit", "Rendered PDF page exceeds the pixel limit.", 413,
                                                       details={"limit": "pixels_per_page", "actual": render_width * render_height,
                                                                "maximum": settings.max_pixels, "width_pixels": render_width,
                                                                "height_pixels": render_height, "render_dpi": 250})
                                bitmap = page.render(scale=scale)
                                try:
                                    with bitmap.to_pil() as image:
                                        with error_context(stage="ocr"):
                                            text = recognize(image, directory, settings, deadline)
                                finally:
                                    bitmap.close()
                                method = "ocr"
                            append(index, number + 1, text.replace("\r\n", "\n"), method)
            except pdfium.PdfiumError as exc:
                raise ServiceError("invalid_document", "PDF is damaged, unsupported, or password protected.",
                                   details={"stage": "pdf_open", "exception_type": type(exc).__name__,
                                            "pdfium_error_code": getattr(exc, "err_code", None),
                                            "hint": "Verify the PDF opens in a viewer and remove password protection before uploading.",
                                            **getattr(exc, "_ocr_context", {})}) from None
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format not in {"PNG", "JPEG", "TIFF", "WEBP"}:
                            raise ServiceError("unsupported_document", "Use PDF, PNG, JPEG, TIFF, or WebP.", 415,
                                               details={"stage": "image_open", "detected_format": image.format})
                        frames = getattr(image, "n_frames", 1)
                        if frames + len(pages) > settings.max_pages:
                            raise ServiceError("page_limit", "Combined page count exceeds the limit.", 413,
                                               details={"limit": "pages", "actual": frames + len(pages), "maximum": settings.max_pages,
                                                        "document_pages": frames, "previous_pages": len(pages)})
                        for frame in range(frames):
                            with error_context(stage="image_page", page=frame + 1):
                                image.seek(frame)
                                if image.width * image.height > settings.max_pixels:
                                    raise ServiceError("pixel_limit", "Image exceeds the pixel limit.", 413,
                                                       details={"limit": "pixels_per_page", "actual": image.width * image.height,
                                                                "maximum": settings.max_pixels, "width_pixels": image.width, "height_pixels": image.height})
                                with ImageOps.exif_transpose(image) as oriented:
                                    with oriented.convert("RGB") as rgb, error_context(stage="ocr"):
                                        text = recognize(rgb, directory, settings, deadline)
                                append(index, frame + 1, text, "ocr")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                raise ServiceError("invalid_document", "Image is invalid or too large.",
                                   details={"stage": "image_open", "exception_type": type(exc).__name__,
                                            "hint": "Verify the image is readable and within the reported pixel limit; re-export corrupt images.",
                                            **getattr(exc, "_ocr_context", {})}) from None
    for index, filename in enumerate(files, 1):
        with error_context(stage="document_read", document=index):
            read_file(index, filename)
    if not any(page["text"].strip() for page in pages):
        raise ServiceError("no_text", "No readable text found; provide a clearer scan.",
                           details={"stage": "document_read", "documents": len(files), "pages_processed": len(pages),
                                    "force_ocr": force_ocr, "hint": "Check scan orientation and resolution and that OCR_LANGUAGE matches the documents."})
    return pages, warnings_list

