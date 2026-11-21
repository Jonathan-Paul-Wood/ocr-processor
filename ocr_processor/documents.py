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

from .errors import ServiceError


def recognize(image, directory, settings, deadline):
    image_path = Path(directory) / "ocr.png"
    output = Path(directory) / "ocr-result"
    image.save(image_path)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ServiceError("timeout", "Document processing exceeded its time limit.", 504)
    try:
        subprocess.run(
            [settings.tesseract_cmd, str(image_path), str(output), "-l", settings.ocr_language, "--psm", "3"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=min(settings.ocr_timeout, remaining),
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
        return output.with_suffix(".txt").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ServiceError("ocr_unavailable", "Tesseract is not installed or configured.", 503) from None
    except subprocess.TimeoutExpired:
        raise ServiceError("timeout", "OCR exceeded its time limit.", 504) from None
    except subprocess.CalledProcessError:
        raise ServiceError("ocr_failed", "OCR failed; check the document and installed language packs.") from None


def read_documents(files, directory, settings, deadline, force_ocr=False):
    pages = []
    chars = 0
    warnings_list = []

    def append(document, number, text, method):
        nonlocal chars
        if len(pages) >= settings.max_pages:
            raise ServiceError("page_limit", "Combined page count exceeds the limit.", 413)
        chars += len(text)
        if chars > settings.max_text_chars:
            raise ServiceError("text_limit", "Extracted text exceeds the limit; split the documents.", 413)
        pages.append({"document": document, "page": number, "method": method, "text": text})
        if not text.strip():
            warnings_list.append(f"Document {document}, page {number} contains no readable text.")

    for index, filename in enumerate(files, 1):
        path = Path(filename)
        with path.open("rb") as source:
            signature = source.read(5)
        if signature == b"%PDF-":
            try:
                with pdfium.PdfDocument(path) as pdf:
                    if len(pdf) == 0 or len(pdf) + len(pages) > settings.max_pages:
                        raise ServiceError("page_limit", "PDF is empty or combined page count exceeds the limit.", 413)
                    for number in range(len(pdf)):
                        with closing(pdf[number]) as page:
                            with closing(page.get_textpage()) as textpage:
                                if textpage.count_chars() > settings.max_text_chars:
                                    raise ServiceError("text_limit", "Page text exceeds the limit.", 413)
                                text = textpage.get_text_bounded().strip()
                            method = "embedded"
                            if force_ocr or len(text) < 40:
                                width, height = page.get_size()
                                scale = 250 / 72
                                if not all(math.isfinite(x) and x > 0 for x in (width, height)) or math.ceil(width * scale) * math.ceil(height * scale) > settings.max_pixels:
                                    raise ServiceError("pixel_limit", "Rendered PDF page exceeds the pixel limit.", 413)
                                bitmap = page.render(scale=scale)
                                try:
                                    with bitmap.to_pil() as image:
                                        text = recognize(image, directory, settings, deadline)
                                finally:
                                    bitmap.close()
                                method = "ocr"
                            append(index, number + 1, text.replace("\r\n", "\n"), method)
            except pdfium.PdfiumError:
                raise ServiceError("invalid_document", "PDF is damaged, unsupported, or password protected.") from None
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format not in {"PNG", "JPEG", "TIFF", "WEBP"}:
                            raise ServiceError("unsupported_document", "Use PDF, PNG, JPEG, TIFF, or WebP.", 415)
                        frames = getattr(image, "n_frames", 1)
                        if frames + len(pages) > settings.max_pages:
                            raise ServiceError("page_limit", "Combined page count exceeds the limit.", 413)
                        for frame in range(frames):
                            image.seek(frame)
                            if image.width * image.height > settings.max_pixels:
                                raise ServiceError("pixel_limit", "Image exceeds the pixel limit.", 413)
                            with ImageOps.exif_transpose(image) as oriented:
                                with oriented.convert("RGB") as rgb:
                                    text = recognize(rgb, directory, settings, deadline)
                            append(index, frame + 1, text, "ocr")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
                raise ServiceError("invalid_document", "Image is invalid or too large.") from None
    if not any(page["text"].strip() for page in pages):
        raise ServiceError("no_text", "No readable text found; provide a clearer scan.")
    return pages, warnings_list

