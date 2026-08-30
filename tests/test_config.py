"""Testes da configuração central."""

from pathlib import Path

from src.config import (
    CATEGORICAL_FEATURES,
    CODE_LIKE_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    Settings,
    get_settings,
    load_params,
)


def test_feature_lists_have_expected_sizes() -> None:
    assert len(NUMERIC_FEATURES) == 10
    assert len(CATEGORICAL_FEATURES) == 7
    assert TARGET == "Revenue"


def test_no_feature_appears_in_both_lists() -> None:
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_code_like_features_are_all_categorical() -> None:
    assert set(CODE_LIKE_FEATURES) <= set(CATEGORICAL_FEATURES)


def test_settings_defaults_are_usable_without_env_file() -> None:
    settings = Settings(_env_file=None)
    assert settings.mlflow_tracking_uri.startswith("sqlite:///")
    assert isinstance(settings.model_dir, Path)


def test_model_uri_points_to_registry_alias() -> None:
    settings = Settings(_env_file=None)
    assert settings.model_uri == (
        f"models:/{settings.mlflow_registered_model}@{settings.mlflow_model_alias}"
    )


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_params_yaml_declares_every_required_section() -> None:
    params = load_params()
    assert {"seed", "data", "split", "train", "evaluate"} <= params.keys()
    assert params["train"]["selection_metric"] == "cv_pr_auc"


def test_every_configured_model_has_hyperparams() -> None:
    train = load_params()["train"]
    assert set(train["models"]) <= set(train["hyperparams"])


def test_request_verify_defaults_to_certifi() -> None:
    assert Settings(_env_file=None).request_verify is True


def test_request_verify_uses_the_configured_bundle() -> None:
    settings = Settings(_env_file=None, ca_bundle=Path("/etc/ssl/corp.pem"))
    assert settings.request_verify == str(Path("/etc/ssl/corp.pem"))
