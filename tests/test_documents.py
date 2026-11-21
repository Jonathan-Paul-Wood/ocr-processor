from dataclasses import replace
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont
import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from ocr_processor import documents
from ocr_processor.documents import read_documents, recognize
from ocr_processor.errors import ServiceError


def test_scanned_pdf_uses_ocr_and_respects_force(settings, tmp_path, monkeypatch, pdf_bytes):
    calls = []

    def ocr(image, *args):
        calls.append(image.size)
        return "Receipt total USD 12.34"

    monkeypatch.setattr(documents, "recognize", ocr)
    pdf = tmp_path / "embedded.pdf"
    pdf.write_bytes(pdf_bytes)
    pages, _ = read_documents([pdf], tmp_path, settings, time.monotonic() + 30, force_ocr=True)
    assert pages[0]["method"] == "ocr"
    assert len(calls) == 1
    scan = tmp_path / "scan.pdf"
    image = Image.new("RGB", (600, 200), "white")
    document = canvas.Canvas(str(scan))
    document.drawImage(ImageReader(image), 20, 600, width=300, height=100)
    document.save()
    pages, _ = read_documents([scan], tmp_path, settings, time.monotonic() + 30)
    assert pages[0]["method"] == "ocr"
    assert len(calls) == 2


def test_multiframe_tiff_and_limits(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "recognize", lambda *args: "Total 12.34")
    path = tmp_path / "receipt.tiff"
    image = Image.new("RGB", (100, 100), "white")
    image.save(path, save_all=True, append_images=[image])
    pages, _ = read_documents([path], tmp_path, settings, time.monotonic() + 30)
    assert [p["page"] for p in pages] == [1, 2]
    for config, code in [(replace(settings, max_pages=1), "page_limit"),
                         (replace(settings, max_pixels=50), "pixel_limit"),
                         (replace(settings, max_text_chars=5), "text_limit")]:
        with pytest.raises(ServiceError) as exc:
            read_documents([path], tmp_path, config, time.monotonic() + 30)
        assert exc.value.code == code


def test_no_text_fails(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "recognize", lambda *args: "")
    path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100), "white").save(path)
    with pytest.raises(ServiceError, match="No readable text"):
        read_documents([path], tmp_path, settings, time.monotonic() + 30)


def test_ocr_timeout_is_explicit(settings, tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        assert kwargs["timeout"] <= settings.ocr_timeout
        raise subprocess.TimeoutExpired("tesseract", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ServiceError) as exc:
        recognize(Image.new("RGB", (10, 10)), tmp_path, settings, time.monotonic() + 5)
    assert exc.value.status == 504


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Native Tesseract is not installed")
def test_native_tesseract_receipt(settings, tmp_path):
    image = Image.new("RGB", (1400, 500), "white")
    font = ImageFont.load_default(size=48)
    draw = ImageDraw.Draw(image)
    draw.text((80, 80), "EXAMPLE MARKET\n2026-11-15\nTOTAL USD 12.34", fill="black", font=font)
    path = tmp_path / "receipt.png"
    image.save(path)
    pages, _ = read_documents([path], tmp_path, settings, time.monotonic() + 30)
    assert "12.34" in pages[0]["text"]

