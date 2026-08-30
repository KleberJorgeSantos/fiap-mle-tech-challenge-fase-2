# ============================================================
# Dockerfile multi-stage — Purchase Intent
# ============================================================
# Dois alvos, um manifesto:
#
#   pipeline (padrão)  executa o projeto de ponta a ponta com `dvc repro`
#   serving            sobe a API que serve o modelo campeão
#
# ── Executar o pipeline (download → preprocess → train → evaluate → register)
#   docker build -t purchase-intent-pipeline .
#   docker run --rm -v "${PWD}/reports:/app/reports" purchase-intent-pipeline
#
# ── Servir o modelo
#   docker build --target serving -t purchase-intent .
#   docker run --rm -p 8000:8000 purchase-intent
#   curl http://localhost:8000/health
# ============================================================


# ── Base comum: Poetry instalado e manifesto copiado ────────
# Os dois builders herdam esta camada, então o download do Poetry e a
# resolução do lock acontecem uma vez só.
FROM python:3.11-slim AS poetry-base

# O venv fica em /opt/app/.venv — fora de /app — para que montar o
# projeto em /app não esconda o interpretador. O caminho precisa ser
# idêntico no builder e no runtime, senão os shebangs do venv quebram.
WORKDIR /opt/app

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

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copiar só o manifesto antes do código faz o Docker reaproveitar esta
# camada enquanto as dependências não mudarem.
COPY pyproject.toml poetry.lock README.md ./


# ── Builder de produção: apenas o grupo main ────────────────
FROM poetry-base AS builder-serving
RUN poetry install --only main --no-root


# ── Builder do pipeline: main + dev (traz DVC e matplotlib) ─
FROM poetry-base AS builder-pipeline
RUN poetry install --with dev --no-root


# ── Alvo `serving`: API enxuta, sem ferramenta de treino ────
FROM python:3.11-slim AS serving

WORKDIR /app

COPY --from=builder-serving /opt/app/.venv /opt/app/.venv

ENV PATH="/opt/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY params.yaml ./

# Artefato produzido pelo pipeline. Se models/ estiver vazio, a API sobe
# assim mesmo: /health responde "degraded" e /predict retorna 503.
COPY models/ ./models/

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]


# ── Alvo `pipeline` (padrão): executa o projeto ─────────────
# Último estágio do arquivo, então `docker build .` sem --target produz
# esta imagem — o `docker run` dela roda o pipeline completo.
FROM python:3.11-slim AS pipeline

WORKDIR /app

COPY --from=builder-pipeline /opt/app/.venv /opt/app/.venv

# A mesma CA da build: o estágio `download` faz HTTPS na UCI em tempo de
# execução, então precisa confiar na raiz da rede também aqui.
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates

ENV PATH="/opt/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY src/ ./src/
COPY dvc.yaml dvc.lock params.yaml ./
COPY .dvc/ ./.dvc/

# O DVC exige um repositório git para operar, e a imagem não carrega o
# .git. `core.no_scm` desliga essa exigência — e como config.local é
# gitignorado, isso vale só dentro do container.
RUN dvc config --local core.no_scm true

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["dvc", "repro"]
