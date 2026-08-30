# Model Card — purchase-intent-classifier

## Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome no Registry | `purchase-intent-classifier` |
| Versão | promovida com o alias `@champion` (veja `reports/registered_model.json`) |
| Algoritmo | `GradientBoostingClassifier` (Scikit-Learn) |
| Artefato | `sklearn.pipeline.Pipeline` — `ColumnTransformer` + classificador |
| Hiperparâmetros | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.9`, `random_state=42` |
| Data do treino | 30/08/2026 |
| Responsável | Grupo 10MLET — FIAP Tech Challenge Fase 2 |

## Uso pretendido

**Para que serve:** priorizar incentivos comerciais durante sessões de
navegação em um e-commerce, estimando a probabilidade de conversão.

**Para que _não_ serve:** decidir preço individual por cliente, negar acesso a
funcionalidades, ou qualquer decisão que afete o usuário negativamente. O
modelo estima intenção agregada de compra, não solvência nem risco.

## Dados de treino

- Online Shoppers Purchasing Intention (UCI id 468) — 12.330 sessões.
- Divisão estratificada 80/20 com `random_state=42`: 9.864 treino / 2.466 teste.
- Taxa de conversão preservada nos dois lados: 15,5%.
- Pré-processamento (mediana + padronização nas numéricas, moda + One-Hot nas
  categóricas) é ajustado **somente no treino**, dentro do `Pipeline`.

## Métricas

Avaliação no conjunto de teste (2.466 sessões nunca vistas), limiar 0,5:

| Métrica | Valor |
|---|---|
| **PR-AUC** (teste) | **0,7391** |
| ROC-AUC | 0,9262 |
| Precision | 0,7068 |
| Recall | 0,5681 |
| F1 | 0,6299 |
| Accuracy | 0,8966 |
| Custo de negócio | R$ 8.700 (90 FP + 165 FN) |

A seleção do campeão usou a **PR-AUC da validação cruzada** (5-fold estratificado no
treino), não a do teste — o teste é reservado para esta medição final. Comparação com os
demais candidatos (`reports/comparison.csv`), ordenada pela métrica de seleção:

| Modelo | CV PR-AUC (seleção) | PR-AUC (teste) | ROC-AUC | Recall |
|---|---:|---:|---:|---:|
| **gradient_boosting** | **0,7380** | 0,7391 | 0,9262 | 0,5681 |
| random_forest | 0,7228 | 0,7079 | 0,9195 | 0,8141 |
| logistic_regression | 0,6571 | 0,6224 | 0,8932 | 0,7435 |
| dummy (baseline) | 0,1547 | 0,1549 | 0,5000 | 0,0000 |

O baseline `DummyClassifier` fixa o piso: PR-AUC de 0,155 é exatamente a taxa
de conversão da base. O campeão entrega **4,8× esse piso**.

## Limitações conhecidas

1. **Recall de 0,57 no limiar padrão.** O `random_forest` alcança 0,81 de
   recall com precisão bem menor. Como o falso negativo custa 10× o falso
   positivo, uma operação real deveria **calibrar o limiar** (parâmetro
   `evaluate.threshold` em `params.yaml`) em vez de aceitar 0,5 — a 0,5 o
   modelo deixa 165 vendas passarem para economizar cupons.
2. **Sem identificação de usuário.** Cada linha é uma sessão isolada; o modelo
   não aprende histórico de compra do cliente.
3. **Dados de um único e-commerce, de um único ano.** Sazonalidade e mix de
   produtos específicos limitam a transferência para outro negócio.
4. **`PageValues` domina o sinal** e é uma métrica derivada do próprio Google
   Analytics — em um sistema real, precisa estar disponível em tempo de
   inferência, o que exige integração com a camada de analytics.

## Considerações éticas

As features são comportamentais e técnicas — não há gênero, idade, renda ou
qualquer atributo protegido. `Region` é um código sem mapeamento geográfico
público, o que reduz o risco de discriminação territorial, mas também impede
auditar viés regional; se o mapeamento existir na operação real, recomenda-se
medir a taxa de aprovação de incentivo por região.

## Reprodução

```bash
poetry install
poetry run dvc repro
```

O `random_state=42` está fixado em `params.yaml` e propagado para o split, a
validação cruzada e o classificador. Reexecuções do pipeline produzem **métricas
idênticas** e geram uma nova versão no Registry, para a qual o alias `@champion`
é reapontado — o histórico de versões anteriores permanece consultável.
