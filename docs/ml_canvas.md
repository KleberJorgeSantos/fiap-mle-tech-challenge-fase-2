# ML Canvas — Propensão de Compra em E-commerce

## 1. Proposta de valor

Identificar, **durante a sessão de navegação**, quais visitantes têm alta
probabilidade de concluir uma compra — para que o time de growth concentre
cupons, frete grátis e retargeting em quem ainda está indeciso, em vez de
distribuir desconto para todo mundo.

Apenas **15,5%** das sessões terminam em compra. Sem modelo, uma campanha
"cupom para todos" queima orçamento em 84,5% dos casos.

## 2. Fonte de dados

| Item | Descrição |
|---|---|
| Dataset | [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468) (UCI, id 468) |
| Volume | 12.330 sessões × 18 colunas |
| Período | 1 ano de operação de um e-commerce real, sem sessões repetidas do mesmo usuário |
| Qualidade | Zero valores ausentes |
| Versionamento | DVC — o hash do CSV está no `dvc.lock` |

## 3. Predição

- **Tipo:** classificação binária.
- **Alvo:** `Revenue` — a sessão terminou em transação?
- **Saída:** probabilidade contínua de conversão + classe no limiar de 0,5.
- **Granularidade:** uma predição por sessão.

## 4. Features

| Grupo | Features | Intuição |
|---|---|---|
| Volume de navegação | `Administrative`, `Informational`, `ProductRelated` | quantas páginas de cada tipo foram vistas |
| Tempo de navegação | `*_Duration` (3 colunas) | engajamento real, não só cliques |
| Qualidade da sessão | `BounceRates`, `ExitRates`, `PageValues` | `PageValues` é a feature mais forte: valor médio das páginas visitadas |
| Contexto temporal | `SpecialDay`, `Month` | proximidade de datas comemorativas e sazonalidade |
| Contexto técnico | `OperatingSystems`, `Browser`, `Region`, `TrafficType` | códigos categóricos, não grandezas |
| Perfil | `VisitorType`, `Weekend` | visitante novo vs. recorrente |

## 5. Decisões orientadas pelo modelo

| Faixa de probabilidade | Ação sugerida |
|---|---|
| ≥ 0,5 | Sessão já quente — não gastar desconto, apenas reduzir atrito no checkout |
| 0,2 – 0,5 | Zona de intervenção — cupom, frete grátis, prova social |
| < 0,2 | Baixo retorno esperado — no máximo retargeting barato |

## 6. Métricas de avaliação

- **Métrica de seleção: PR-AUC.** Com 15,5% de positivos, a accuracy é enganosa
  — um modelo que sempre responde "não compra" já acerta 84,5%.
- **Métricas de apoio:** ROC-AUC, precision, recall, F1.
- **Métrica de negócio:** custo assimétrico dos erros.

| Erro | Significado | Custo unitário |
|---|---|---|
| Falso Positivo | Cupom dado a quem já ia comprar / não ia comprar de todo jeito | R$ 5 |
| Falso Negativo | Sessão que ia converter e não recebeu incentivo — venda perdida | R$ 50 |

## 7. Modelo em produção

- Artefato: `Pipeline` do Scikit-Learn (pré-processamento + classificador) em
  um único objeto, servido pela API FastAPI.
- Registro: MLflow Model Registry, alias `@champion`.
- Fallback: se o Registry estiver indisponível, a API carrega
  `models/model.joblib` gerado pelo `dvc repro`.

## 8. Monitoramento e riscos

| Risco | Sinal de alerta | Mitigação |
|---|---|---|
| Drift sazonal | distribuição de `Month` muda | retreinar a cada trimestre |
| Mudança no layout do site | `PageValues` e `ExitRates` deslocam | acompanhar a distribuição das features |
| Códigos novos (browser, região) | categorias não vistas no treino | `OneHotEncoder(handle_unknown="ignore")` já absorve |
| Feedback loop | dar cupom altera o comportamento medido | manter um grupo de controle sem intervenção |
