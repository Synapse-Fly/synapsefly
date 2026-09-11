# SynapseFly backend (the brain). Build context = repo root:
#   docker build -f backend/Dockerfile -t synapsefly-backend .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    FLY_DATA_DIR=/app/data \
    FLY_OUT_DIR=/app/out \
    FLY_HOST=0.0.0.0 \
    FLY_PORT=4000

WORKDIR /app

# Deps first for layer caching.
COPY deploy/requirements.runtime.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# App code — only the package + entrypoint (NOT tests/, data/, out/, so nothing local leaks in).
COPY backend/flybrain/ /app/backend/flybrain/
COPY backend/run.py /app/backend/run.py
RUN mkdir -p /app/data /app/out

EXPOSE 4000

# Simple in-image healthcheck against the FastAPI /api/health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"ok\":true' in urllib.request.urlopen('http://127.0.0.1:4000/api/health',timeout=4).read() else 1)"

CMD ["python", "/app/backend/run.py"]
