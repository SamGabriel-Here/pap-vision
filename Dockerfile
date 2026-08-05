FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py preprocessing.py cervical_model.pth model_metadata.json ./
COPY templates/ templates/

RUN useradd --create-home --uid 1000 papvision \
    && chown -R papvision:papvision /app
USER papvision

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)" || exit 1

# 2 workers, matching the README's suggested production command. With the
# default in-memory rate-limit storage (see PAPVISION_RATE_LIMIT_STORAGE_URI
# in app.py) each worker keeps its own counter, so the effective /predict
# ceiling under concurrent load is up to (workers x PAPVISION_PREDICT_RATE_LIMIT).
# Point PAPVISION_RATE_LIMIT_STORAGE_URI at Redis for an exact shared limit.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", "app:app"]
