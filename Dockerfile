# ShieldML Production Multi-Stage Dockerfile
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.10-slim AS runtime

WORKDIR /app

# Create non-root user
RUN groupadd -r shieldml && useradd -r -g shieldml -s /bin/false shieldml

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY pyproject.toml ./

RUN pip install --no-deps -e .

RUN chown -R shieldml:shieldml /app
USER shieldml

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
