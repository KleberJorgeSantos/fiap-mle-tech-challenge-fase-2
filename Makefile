.PHONY: help install lint format test repro dag metrics push pull mlflow api clean \
        docker-pipeline docker-build docker-run

POETRY := poetry run

help:           ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Ambiente ─────────────────────────────────────────────────────────────────

install:        ## Instala todas as dependências (prod + dev)
	poetry install

# ── Qualidade de código ──────────────────────────────────────────────────────

lint:           ## Verifica estilo e imports com Ruff
	$(POETRY) ruff check src tests

format:         ## Formata o código com Ruff
	$(POETRY) ruff format src tests

test:           ## Roda a suíte de testes com cobertura
	$(POETRY) pytest

# ── Pipeline DVC ─────────────────────────────────────────────────────────────

repro:          ## Executa o pipeline completo (download → register)
	$(POETRY) dvc repro

dag:            ## Desenha o grafo de estágios
	$(POETRY) dvc dag

metrics:        ## Mostra as métricas do último run
	$(POETRY) dvc metrics show

push:           ## Envia dados e artefatos para o remote do DVC
	$(POETRY) dvc push

pull:           ## Recupera dados e artefatos do remote do DVC
	$(POETRY) dvc pull

# ── MLflow e API ─────────────────────────────────────────────────────────────

mlflow:         ## Abre a UI do MLflow em http://localhost:5000
	$(POETRY) mlflow ui --backend-store-uri sqlite:///mlflow.db

api:            ## Sobe a API local em http://localhost:8000/docs
	$(POETRY) uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker ───────────────────────────────────────────────────────────────────

docker-pipeline: ## Constrói e EXECUTA o pipeline completo em container
	docker build -t purchase-intent-pipeline .
	docker run --rm purchase-intent-pipeline

docker-build:   ## Constrói a imagem de serving (API)
	docker build --target serving -t purchase-intent .

docker-run:     ## Sobe a API containerizada na porta 8000
	docker run --rm -p 8000:8000 purchase-intent

# ── Limpeza ──────────────────────────────────────────────────────────────────

clean:          ## Remove caches de testes e lint
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
