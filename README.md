<div align="center">

# 🛒 Purchase Intent

### Propensão de compra em e-commerce — FIAP Tech Challenge · Fase 2

Pipeline de Machine Learning **reprodutível de ponta a ponta**: dados versionados com DVC,
experimentos rastreados no MLflow, modelo promovido no Model Registry e API containerizada.

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-2.4-60A5FA?style=for-the-badge&logo=poetry&logoColor=white)](https://python-poetry.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.22-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.67-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff)
[![Tests](https://img.shields.io/badge/Testes-73%20·%2088%25-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![GitHub](https://img.shields.io/badge/GitHub-Reposit%C3%B3rio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KleberJorgeSantos/fiap-mle-tech-challenge-fase-2)

</div>

---

## 📋 Sumário

- [Contexto](#-contexto)
- [Resultados](#-resultados)
- [Arquitetura](#-arquitetura)
- [Quick Start](#-quick-start)
- [Pipeline DVC](#-pipeline-dvc)
- [MLflow e Model Registry](#-mlflow-e-model-registry)
- [API](#-api)
- [Docker](#-docker)
- [Configuração](#-configuração)
- [Qualidade de código](#-qualidade-de-código)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Como este projeto atende ao desafio](#-como-este-projeto-atende-ao-desafio)

---

## 🎯 Contexto

Uma empresa de e-commerce precisa saber, **durante a sessão de navegação**, quais visitantes
têm propensão a comprar — para concentrar cupons e retargeting em quem ainda está indeciso,
em vez de distribuir desconto para todo mundo.

| | |
|---|---|
| 📦 **Dataset** | [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468) (UCI id 468) — 12.330 sessões, 17 features |
| 🎯 **Alvo** | `Revenue` — a sessão terminou em transação? |
| 📊 **Taxa de conversão** | **15,5%** — classes bastante desbalanceadas |
| 💰 **Custo de negócio** | FP = R$ 5 (cupom desperdiçado) · FN = R$ 50 (venda perdida) |

> O foco do desafio é **engenharia de ML**, não a complexidade do modelo. O que este
> repositório demonstra é o caminho completo: dado versionado → pipeline reprodutível →
> experimento rastreado → modelo promovido → API servindo.

---

## 🏆 Resultados

A métrica de seleção é a **PR-AUC da validação cruzada**, não a accuracy. Com 15,5% de
positivos, um classificador que responde "ninguém compra" já acerta **84,5%** — e é exatamente
isso que o baseline `DummyClassifier` prova na tabela abaixo.

| Modelo | CV PR-AUC ⭐ | PR-AUC (teste) | ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Gradient Boosting** — *servido* | **0,7380** | 0,7391 | 0,9262 | 0,7068 | 0,5681 | 0,6299 | 0,8966 |
| 🥈 Random Forest | 0,7228 | 0,7079 | 0,9195 | 0,5353 | 0,8141 | 0,6459 | 0,8617 |
| 🥉 Logistic Regression | 0,6571 | 0,6224 | 0,8932 | 0,4913 | 0,7435 | 0,5917 | 0,8410 |
| 🎲 Dummy (baseline) | 0,1547 | 0,1549 | 0,5000 | 0,0000 | 0,0000 | 0,0000 | **0,8451** |

> **Por que a coluna de seleção é a CV, e não a do teste?** Escolher o modelo olhando o
> conjunto de teste transformaria o teste em parte da decisão — e a métrica final deixaria de
> ser uma estimativa imparcial. A seleção usa 5-fold estratificado **dentro do treino**; o
> teste é tocado uma única vez, no estágio `evaluate`.

<div align="center">

**O baseline acerta 84,5% das sessões e não vale absolutamente nada.**
É por isso que a accuracy não elege o campeão aqui.

</div>

O Gradient Boosting entrega **4,8× a PR-AUC do baseline** (0,7391 contra o piso de 0,155, que
é a própria taxa de conversão da base). No limiar 0,5 isso significa 90 falsos positivos e
165 falsos negativos — **R$ 8.700** de custo de negócio no conjunto de teste.

> ⚠️ **Nota honesta:** com o falso negativo custando 10× o falso positivo, o limiar 0,5
> provavelmente não é o ponto ótimo — o Random Forest tem recall de 0,81 contra 0,57 do
> campeão. Calibrar `evaluate.threshold` no `params.yaml` é o próximo passo natural, e está
> registrado como limitação em [`docs/model_card.md`](docs/model_card.md).

Métricas geradas por `dvc repro` → `reports/metrics.json` e `reports/comparison.csv`.

---

## 🏗️ Arquitetura

```
data/raw/online_shoppers_intention.csv          [DVC · estagio download]
  └─ src/data/loader.py            load_raw() → get_features_target() → validate_schema()
  │                                [Pandera valida faixas; codigos int → str]
  └─ src/data/preprocessing.py     build_preprocessor()  [ColumnTransformer → ~75 features]
  │                                split_data()          [estratificado, seed 42]
  └─ src/models/factory.py         build_model_pipeline() [pre-processador + classificador]
  └─ src/models/train.py           train_all_models()     [StratifiedKFold 5-fold]
  └─ src/evaluation/metrics.py     evaluate_model() · cost_analysis()
  └─ src/tracking/mlflow_utils.py  log_model_run()        [1 run por candidato]
  └─ src/tracking/registry.py      register_best_model()  [Registry + alias @champion]
       ↓
  src/api/main.py                  models:/purchase-intent-classifier@champion
```

**Duas decisões de arquitetura que carregam o projeto:**

1. **Pré-processamento e classificador vivem no mesmo `Pipeline`.** O artefato servido recebe
   o DataFrame **cru** da sessão e cuida sozinho da padronização e do One-Hot. Isso elimina a
   classe de bug mais comum em produção — treino e inferência divergirem no pré-processo.

2. **O `ColumnTransformer` é ajustado apenas no treino.** Como ele está dentro do `Pipeline`,
   tanto a validação cruzada quanto o `fit` final respeitam a separação automaticamente:
   não há como as estatísticas do teste vazarem para o modelo.

---

## 🚀 Quick Start

**Pré-requisitos:** Python 3.11 e [Poetry](https://python-poetry.org/docs/#installation) 2.x.

```bash
git clone https://github.com/KleberJorgeSantos/fiap-mle-tech-challenge-fase-2.git
cd fiap-mle-tech-challenge-fase-2

cp .env.example .env          # ajuste se necessario; os defaults ja funcionam
poetry install                # instala prod + dev a partir do poetry.lock

poetry run dvc repro          # pipeline completo: download → preprocess → train → evaluate → register
poetry run dvc metrics show   # metricas do modelo campeao
```

Pronto. O `dvc repro` baixa o dataset, treina os quatro candidatos, rastreia tudo no MLflow,
promove o vencedor no Registry e salva `models/model.joblib`.

<details>
<summary><b>Atalhos do Makefile</b></summary>

```bash
make help          # lista todos os alvos
make install       # poetry install
make lint          # ruff check
make format        # ruff format
make test          # pytest com cobertura
make repro         # dvc repro
make dag           # desenha o grafo de estagios
make metrics       # dvc metrics show
make push / pull   # sincroniza com o remote do DVC
make mlflow        # UI do MLflow em :5000
make api           # API local em :8000
make docker-build  # constroi a imagem
make docker-run    # sobe a API containerizada
```
</details>

---

## 🔁 Pipeline DVC

Cinco estágios encadeados em [`dvc.yaml`](dvc.yaml). O DVC compara os hashes de `deps`,
`params` e `outs` e **só re-executa o que mudou**.

```
+----------+
| download |   baixa o zip da UCI e extrai o CSV
+----------+
     |
+------------+
| preprocess |  valida schema (Pandera), tipa codigos, divide 80/20 estratificado
+------------+
     |
  +--+---------------+
+-------+        +----------+
| train |        | evaluate |  metricas + curvas ROC/PR
+-------+        +----------+
     |
+----------+
| register |  promove o campeao no Model Registry com o alias @champion
+----------+
```

| Estágio | Entradas rastreadas | Saídas |
|---|---|---|
| `download` | `data.url`, `data.csv_name` | `data/raw/*.csv` |
| `preprocess` | CSV, `seed`, `split` | `data/processed/{train,test}.parquet` |
| `train` | parquets, `seed`, `train` | `models/model.joblib`, `reports/comparison.csv`, `reports/best_run.json` |
| `evaluate` | modelo, teste, `evaluate` | `reports/metrics.json`, `reports/figures/` |
| `register` | `reports/best_run.json` | `reports/registered_model.json` |

### Versionamento de dados

O dataset **não está no git** — o que está versionado é o hash, dentro do `dvc.lock`. É esse
hash que garante que todo mundo treina exatamente sobre os mesmos 12.330 registros: se o
arquivo da UCI mudar amanhã, o `dvc repro` acusa a diferença em vez de treinar em silêncio
sobre outros dados.

Os bytes ficam no **remote**, configurado em `.dvc/config` como um diretório local
(`../tc2-dvc-storage`) — escolha deliberada para que o projeto rode sem credenciais.

```bash
poetry run dvc push           # envia dados e artefatos para o remote
rm -rf data/ models/          # apaga tudo localmente
poetry run dvc pull           # recupera exatamente os mesmos bytes
```

> **Clonando do GitHub?** O remote local existe apenas na máquina que rodou o `dvc push`, então
> `dvc pull` não vai encontrar nada em um clone novo — e não precisa: rode **`dvc repro`**, que
> baixa o dataset da UCI no estágio `download` e reconstrói tudo do zero. Em um time de
> verdade, bastaria trocar uma linha para um bucket compartilhado:
>
> ```bash
> dvc remote add -d storage s3://meu-bucket/purchase-intent
> ```

### Comandos úteis

```bash
poetry run dvc dag                    # grafo de estagios
poetry run dvc status                 # o que esta desatualizado
poetry run dvc metrics show           # metricas atuais
poetry run dvc params diff            # o que mudou no params.yaml
poetry run dvc repro --force train    # forca a re-execucao de um estagio
```

### Experimentar hiperparâmetros

Tudo que define o experimento está em [`params.yaml`](params.yaml). Altere um valor e rode
`dvc repro`: só os estágios afetados re-executam.

```yaml
train:
  selection_metric: pr_auc      # troque para "recall" e o campeao muda
  hyperparams:
    gradient_boosting:
      n_estimators: 200
      learning_rate: 0.1
```

---

## 📊 MLflow e Model Registry

```bash
make mlflow      # http://localhost:5000
```

**Aba Experiments** → experimento `purchase-intent`, com **um run por candidato** contendo
hiperparâmetros, métricas de teste, a PR-AUC da validação cruzada e o modelo serializado com
`signature` e `input_example`.

**Aba Models** → `purchase-intent-classifier`, versão 1, alias **`@champion`**, com tags de
proveniência (`selection_metric`, `selection_value`, `source_run_id`).

> ### Por que alias e não `stage="Production"`?
> Os *stages* do Model Registry foram **depreciados no MLflow 2.9** em favor de **aliases**.
> Um alias é um ponteiro móvel para uma versão específica: a API carrega
> `models:/purchase-intent-classifier@champion` e passa a servir uma nova versão assim que o
> alias for reapontado — **sem nenhuma alteração de código nem novo deploy**.

O backend é **SQLite** (`sqlite:///mlflow.db`). Isso não é detalhe: o Model Registry **não
funciona** com o store baseado em arquivos (`./mlruns`) — ele exige um banco.

---

## 🔌 API

```bash
make api          # http://localhost:8000/docs
```

A API carrega o modelo campeão do Registry e, se o MLflow não estiver acessível, cai para
`models/model.joblib`. Isso mantém a imagem Docker utilizável mesmo sem servidor de tracking.

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Estado do serviço e origem do modelo carregado |
| `POST` | `/predict` | Probabilidade de conversão da sessão |
| `GET` | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/session.json
```

```json
{
  "purchase_probability": 0.8571,
  "will_purchase": true,
  "threshold": 0.5
}
```

**Códigos de erro:** `422` para payload fora do contrato (Pydantic) ou fora das faixas válidas
(Pandera) · `503` quando nenhum modelo pôde ser carregado.

---

## 🐳 Docker

Build **multi-stage**: o Poetry e o cache de wheels ficam no estágio de build e nunca chegam
à imagem final, que roda como usuário **não-root** e expõe um `HEALTHCHECK`.

```bash
make docker-build
make docker-run
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_source":"local-joblib"}
```

Esta é uma imagem de **serving**, não de treino: instala apenas o grupo de produção
(`poetry install --only main`), sem pytest, ruff, DVC ou matplotlib. O pipeline roda no
ambiente de desenvolvimento, com `poetry run dvc repro`.

Duas decisões que enxugaram a imagem de 1,97 GB para **1,49 GB**:

- **`mlflow-skinny` em produção.** A API só precisa do *cliente* para carregar o modelo do
  Registry — servidor, UI, Flask e gunicorn ficam no grupo dev, onde a `mlflow ui` roda.
- **`matplotlib` só em dev.** As curvas ROC/PR são geradas pelo estágio `evaluate`, que nunca
  executa dentro do container de serving.

<details>
<summary><b>Build atrás de proxy corporativo ou antivírus com inspeção TLS</b></summary>

Se o `pip` falhar no build com `certificate verify failed`, sua rede reassina o tráfego HTTPS
com uma CA própria. Coloque o certificado raiz em `certs/` com extensão `.crt` — o Dockerfile
o instala como raiz confiável antes de baixar qualquer dependência. Com o diretório vazio (o
caso normal) o build não muda em nada. Detalhes em [`certs/README.md`](certs/README.md).
</details>

---

## ⚙️ Configuração

Duas fontes, com responsabilidades separadas de propósito:

| Fonte | Contém | Versionado? |
|---|---|---|
| `.env` | Caminhos, URIs, portas — **muda entre máquinas** | ❌ apenas o `.env.example` |
| `params.yaml` | Seed e hiperparâmetros — **define o experimento** | ✅ rastreado pelo DVC |

```bash
cp .env.example .env
```

As variáveis são lidas por `Settings(BaseSettings)` em [`src/config.py`](src/config.py) com
*pydantic-settings*, que valida os tipos na subida. Os defaults já funcionam para
desenvolvimento local — o `.env` só é necessário para apontar para outro servidor MLflow ou
para ajustar a rede.

<details>
<summary><b>Atrás de proxy corporativo ou antivírus com inspeção TLS</b></summary>

Se o estágio `download` falhar com `CERTIFICATE_VERIFY_FAILED`, sua rede reassina o HTTPS com
uma CA que o `certifi` não conhece. Aponte o `.env` para um bundle PEM que inclua essa raiz:

```bash
CA_BUNDLE=C:/caminho/para/ca-bundle.pem
```

O valor chega ao `requests` como `verify=` via `Settings.request_verify`. O mesmo certificado,
com extensão `.crt` em [`certs/`](certs/README.md), resolve o build da imagem Docker.
</details>

---

## ✅ Qualidade de código

```bash
make lint      # ruff check src tests    → All checks passed!
make format    # ruff format src tests
make test      # 73 testes · 88% de cobertura
```

- **Type hints** em todas as funções públicas, com docstrings no estilo Google.
- **Funções curtas** — nenhuma passa de 20 linhas.
- **Zero `print()`**: todo output passa por `logging`, com nível controlado via `LOG_LEVEL`.
- **Ruff** configurado com `E, W, F, I, N, UP, B` (erros, imports, naming, modernização, bugs).
- **Seeds fixados** em `params.yaml` e propagados para split, validação cruzada e classificador.
- A suíte roda **offline**: as fixtures geram um DataFrame sintético com o mesmo esquema do
  CSV da UCI, então nenhum teste depende de download ou de artefato treinado.
- A análise exploratória que justifica essas decisões está em
  [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), executada e com os gráficos salvos.

<details>
<summary><b>Cobertura por módulo</b></summary>

```
src/api/main.py                93%
src/api/schemas.py            100%
src/config.py                 100%
src/data/loader.py            100%
src/data/preprocessing.py      96%
src/evaluation/metrics.py     100%
src/models/factory.py         100%
src/tracking/registry.py      100%
---------------------------------
TOTAL                          88%
```
</details>

---

## 📁 Estrutura do projeto

```
tech_challenge_2/
├── src/
│   ├── config.py               Settings (pydantic-settings) + esquema de features
│   ├── logging_config.py       logging centralizado (sem print)
│   ├── data/
│   │   ├── download.py         baixa e extrai o zip da UCI
│   │   ├── loader.py           leitura, tipagem e validacao Pandera
│   │   └── preprocessing.py    ColumnTransformer + split estratificado
│   ├── models/
│   │   ├── factory.py          MODEL_FACTORY: nome → estimador
│   │   └── train.py            validacao cruzada, treino e selecao
│   ├── evaluation/metrics.py   metricas + custo de negocio
│   ├── tracking/
│   │   ├── mlflow_utils.py     setup do tracking e log de runs
│   │   └── registry.py         registro e promocao via alias
│   ├── pipelines/              entrypoints dos 5 estagios do DVC
│   └── api/                    FastAPI (main.py, schemas.py)
├── tests/                      73 testes, fixtures offline
├── docs/
│   ├── ml_canvas.md            problema, features, decisoes, riscos
│   └── model_card.md           metricas, limitacoes, consideracoes eticas
├── examples/session.json       payload de exemplo para o /predict
├── notebooks/01_eda.ipynb      EDA executada — grafico do desbalanceamento, codigos
│                               categoricos, sazonalidade
├── certs/                      CAs extras para build atras de inspecao TLS (opcional)
├── params.yaml                 seed e hiperparametros (rastreados pelo DVC)
├── dvc.yaml / dvc.lock         definicao e estado do pipeline
├── pyproject.toml              Poetry — prod e dev separados
├── poetry.lock                 versoes travadas (commitado)
├── Dockerfile                  multi-stage, non-root, healthcheck
├── Makefile                    atalhos de desenvolvimento
└── .env.example                template de configuracao
```

---

## 🎓 Como este projeto atende ao desafio

| Critério | Peso | Onde está |
|---|:---:|---|
| **Clean Code e Estrutura** | 20% | Módulos curtos e coesos em `src/`, type hints e docstrings em todas as funções públicas, `MODEL_FACTORY` desacoplando a criação de modelos, Ruff sem apontamentos |
| **Reprodutibilidade** | 20% | `pyproject.toml` com grupos prod/dev separados, `poetry.lock` commitado, `.env.example` no repositório e `.env` ignorado, seeds fixados em `params.yaml` |
| **Docker** | 15% | `Dockerfile` multi-stage instalando via Poetry, usuário não-root, `HEALTHCHECK`, `.dockerignore` completo |
| **DVC + Pipeline** | 15% | `dvc.yaml` com 5 estágios encadeados, dataset versionado por hash no `dvc.lock`, remote configurado, `dvc repro` roda do zero |
| **Modelagem Clássica** | 10% | Quatro candidatos Scikit-Learn com validação cruzada estratificada, incluindo `DummyClassifier` como piso de comparação |
| **MLflow + Registry** | 20% | Um run por candidato com params, métricas e modelo assinado; campeão registrado e promovido com o alias `@champion` |

**Entregáveis:** repositório GitHub · vídeo STAR de 5 min.

---

<div align="center">

**FIAP · Machine Learning Engineering · Tech Challenge Fase 2**

</div>
