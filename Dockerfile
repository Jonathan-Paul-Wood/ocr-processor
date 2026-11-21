FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_THREAD_LIMIT=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home processor
WORKDIR /app
COPY pyproject.toml ./
COPY ocr_processor ./ocr_processor
RUN pip install --no-cache-dir .
USER 10001:10001
EXPOSE 8090
CMD ["python", "-m", "ocr_processor"]

FROM runtime AS test
USER root
RUN pip install --no-cache-dir '.[test]'
COPY tests ./tests
COPY examples ./examples
USER 10001:10001
RUN python -m pytest -q -p no:cacheprovider

FROM runtime AS production

