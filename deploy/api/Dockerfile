# Backend: Flask API served by gunicorn. Build from the repo root:
#   docker build -f deploy/api/Dockerfile -t superkart-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PORT=7860 \
    WEB_CONCURRENCY=2

# libgomp is required by XGBoost/scikit-learn at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY deploy/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser src/ src/
COPY --chown=appuser api/ api/
COPY --chown=appuser models/ models/

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "gunicorn 'api.app:create_app()' --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --access-logfile -"]
