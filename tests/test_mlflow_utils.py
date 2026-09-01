"""Testes da camada fina sobre o MLflow, com o módulo ``mlflow`` mockado.

Nenhum teste aqui precisa de um servidor de tracking real — o objetivo é
verificar que as chamadas certas são feitas com os argumentos certos.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.tracking import mlflow_utils


@patch("src.tracking.mlflow_utils.setup_logging")
@patch("src.tracking.mlflow_utils.mlflow")
def test_setup_tracking_configures_uri_and_experiment(
    mock_mlflow: MagicMock, mock_setup_logging: MagicMock
) -> None:
    mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="42")

    experiment_id = mlflow_utils.setup_tracking()

    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once()
    assert experiment_id == "42"


@patch("src.tracking.mlflow_utils.setup_logging")
@patch("src.tracking.mlflow_utils.mlflow")
def test_setup_tracking_reapplies_logging_after_mlflow_reconfigures_it(
    mock_mlflow: MagicMock, mock_setup_logging: MagicMock
) -> None:
    """O MLflow troca o handler de log para stderr e sobe o nível — sem
    reaplicar nossa config, o `dvc repro` roda mudo."""
    mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="1")

    mlflow_utils.setup_tracking()

    mock_setup_logging.assert_called_once()


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_model_run_logs_params_metrics_and_model(mock_mlflow: MagicMock) -> None:
    run = MagicMock()
    run.info.run_id = "run-123"
    mock_mlflow.start_run.return_value.__enter__.return_value = run

    model = MagicMock()
    model.predict_proba.return_value = [[0.2, 0.8]]
    input_example = pd.DataFrame({"a": [1]})

    run_id = mlflow_utils.log_model_run(
        run_name="modelo-x",
        params={"a": 1},
        metrics={"pr_auc": 0.5},
        model=model,
        input_example=input_example,
    )

    assert run_id == "run-123"
    mock_mlflow.log_params.assert_called_once_with({"a": 1})
    mock_mlflow.log_metrics.assert_called_once_with({"pr_auc": 0.5})
    mock_mlflow.sklearn.log_model.assert_called_once()


@patch("src.tracking.mlflow_utils.mlflow")
def test_log_model_run_opens_the_run_with_the_given_name(mock_mlflow: MagicMock) -> None:
    run = MagicMock()
    run.info.run_id = "run-456"
    mock_mlflow.start_run.return_value.__enter__.return_value = run
    model = MagicMock()
    model.predict_proba.return_value = [[0.5, 0.5]]

    mlflow_utils.log_model_run(
        run_name="gradient_boosting",
        params={},
        metrics={},
        model=model,
        input_example=pd.DataFrame({"a": [1]}),
    )

    mock_mlflow.start_run.assert_called_once_with(run_name="gradient_boosting")
