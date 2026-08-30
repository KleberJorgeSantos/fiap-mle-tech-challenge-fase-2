# ============================================================
# Dockerfile — Purchase Intent API
# ============================================================
# Build multi-stage: o Poetry e o cache de wheels ficam no stage
# de build e nunca chegam à imagem final.
#
#   docker build -t purchase-intent .
#   docker run --rm -p 8000:8000 purchase-intent
#   curl http://localhost:8000/health
#
# Esta é uma imagem de SERVING: instala apenas as dependências de
# produção (`--only main`), sem pytest, ruff, DVC ou matplotlib.
# O pipeline de treino roda no ambiente de desenvolvimento
# (`poetry run dvc repro`), não aqui.
# ============================================================

# ── Stage 1: builder — resolve e instala as dependências ────
FROM python:3.11-slim AS builder

# O WORKDIR precisa ser /app (o mesmo do runtime) para que os
# shebangs dos scripts do venv apontem para o caminho correto.
WORKDIR /app

ENV POETRY_VERSION=2.4.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

# Redes com inspeção TLS (proxy corporativo, antivírus) reassinam o tráfego
# do PyPI com uma CA própria, desconhecida dentro do container. Qualquer .crt
# colocado em certs/ vira raiz confiável aqui. Com o diretório vazio — o caso
# normal — este passo não muda nada.
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt     REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copiar apenas o manifesto e o lock antes do código-fonte faz o
# Docker reaproveitar esta camada enquanto as dependências não mudam.
COPY pyproject.toml poetry.lock README.md ./

# --only main: nada de pytest, ruff ou DVC na imagem de produção.
# --no-root: o projeto é copiado no stage seguinte.
RUN poetry install --only main --no-root

# ── Stage 2: runtime — imagem final enxuta ──────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY params.yaml ./

# Artefato treinado pelo `dvc repro`. Se models/ estiver vazio, a API
# sobe assim mesmo: /health responde "degraded" e /predict retorna 503.
COPY models/ ./models/

# Usuário não-root — a aplicação nunca precisa de privilégios.
RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
