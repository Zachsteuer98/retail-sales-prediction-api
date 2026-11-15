# Frontend: Streamlit demo with no ML dependencies; all inference goes through the API.
#   docker build -f deploy/app/Dockerfile -t retail-sales-app .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=7860

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY deploy/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser src/ src/
COPY --chown=appuser app/ app/
COPY --chown=appuser data/sample/ data/sample/
COPY --chown=appuser reports/ reports/

USER appuser
EXPOSE 7860
# API_URL must point at the backend, e.g. https://<user>-retail-sales-api.hf.space
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]
