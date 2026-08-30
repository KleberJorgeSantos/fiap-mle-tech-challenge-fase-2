"""Testes da promoção no Model Registry, com o MLflow mockado.

Não há servidor de tracking na suíte: o objetivo é garantir que o alias
seja apontado para a versão certa e que as tags de proveniência sejam
gravadas, não exercitar o MLflow em si.
"""

from unittest.mock import MagicMock, patch

from src.tracking import registry


@patch("src.tracking.registry.mlflow.register_model")
def test_register_model_returns_the_new_version(mock_register: MagicMock) -> None:
    mock_register.return_value = MagicMock(version="3")
    assert registry.register_model("run-abc", "meu-modelo") == "3"
    mock_register.assert_called_once_with(model_uri="runs:/run-abc/model", name="meu-modelo")


@patch("src.tracking.registry.MlflowClient")
def test_promote_sets_alias_and_tags(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    registry.promote_to_alias("m", "2", "champion", tags={"k": "v"})

    client.set_registered_model_alias.assert_called_once_with(
        name="m", alias="champion", version="2"
    )
    client.set_model_version_tag.assert_called_once_with(name="m", version="2", key="k", value="v")


@patch("src.tracking.registry.MlflowClient")
def test_promote_without_tags_is_allowed(mock_client_cls: MagicMock) -> None:
    registry.promote_to_alias("m", "1", "champion")
    mock_client_cls.return_value.set_model_version_tag.assert_not_called()


@patch("src.tracking.registry.MlflowClient")
@patch("src.tracking.registry.mlflow.register_model")
def test_register_best_model_summary_is_serializable(
    mock_register: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_register.return_value = MagicMock(version="7")
    summary = registry.register_best_model("run-xyz", "pr_auc", 0.7391)

    assert summary["version"] == "7"
    assert summary["run_id"] == "run-xyz"
    assert summary["selection_metric"] == "pr_auc"
    assert summary["model_uri"].endswith("@champion")
    assert all(isinstance(value, str) for value in summary.values())
