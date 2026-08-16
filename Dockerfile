FROM python:3.11-slim

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Solo lo necesario para servir: código de la API y artefactos ya entrenados.
# training/ y data/ se quedan fuera de la imagen (ver .dockerignore).
COPY app/ ./app/
COPY models/ ./models/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
