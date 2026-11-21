import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    timeout: int = 180
    max_upload_bytes: int = 20 * 1024 * 1024
    max_files: int = 5
    max_pages: int = 30
    max_pixels: int = 16_000_000
    max_text_chars: int = 100_000
    ocr_timeout: int = 30
    ocr_language: str = "eng"
    tesseract_cmd: str = "tesseract"

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("OCR_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180")),
            ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
            tesseract_cmd=os.getenv("TESSERACT_CMD", "tesseract"),
        )

