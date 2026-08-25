# Saathi — voice companion server.
#
# WebRTC media needs direct UDP, so run with host networking (Linux hosts):
#   docker build -t saathi .
#   docker run --network host --env-file /etc/saathi/env saathi
#
# The KB index and the embedding model are baked at build time so cold starts
# are fast and the container needs no outbound HuggingFace access at runtime.

FROM python:3.11-slim

WORKDIR /app

# onnxruntime needs libgomp; everything else is wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/.fastembed_cache \
    SAATHI_HOST=0.0.0.0 \
    SAATHI_PORT=7860

# Classics raw texts are optional at runtime (the curated .md files are in the
# repo); fetch is best-effort so an offline build still succeeds.
RUN python scripts/fetch_classics.py || true
RUN python scripts/build_kb.py

EXPOSE 7860
CMD ["python", "server.py"]
