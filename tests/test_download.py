"""Testes do download do dataset, com ``requests`` mockado.

Nenhum teste aqui toca a rede — o objetivo é a lógica de extração e
gravação do arquivo, não validar que a UCI está no ar.
"""

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.download import download_dataset


def _fake_zip_bytes(csv_name: str, content: bytes) -> bytes:
    """Monta um zip em memória contendo um único arquivo."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(csv_name, content)
    return buffer.getvalue()


@patch("src.data.download.requests.get")
def test_extracts_the_named_csv_from_the_zip(mock_get: MagicMock, tmp_path: Path) -> None:
    mock_get.return_value = MagicMock(content=_fake_zip_bytes("dados.csv", b"a,b\n1,2\n"))
    destination = tmp_path / "raw" / "dados.csv"

    result = download_dataset("http://exemplo/data.zip", destination, "dados.csv")

    assert result == destination
    assert destination.read_bytes() == b"a,b\n1,2\n"


@patch("src.data.download.requests.get")
def test_creates_the_parent_directory_if_missing(mock_get: MagicMock, tmp_path: Path) -> None:
    mock_get.return_value = MagicMock(content=_fake_zip_bytes("d.csv", b"x"))
    destination = tmp_path / "nested" / "dir" / "d.csv"

    download_dataset("http://exemplo/data.zip", destination, "d.csv")

    assert destination.parent.is_dir()


@patch("src.data.download.requests.get")
def test_raises_when_the_http_response_is_an_error(mock_get: MagicMock, tmp_path: Path) -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = RuntimeError("HTTP 404")
    mock_get.return_value = response

    with pytest.raises(RuntimeError, match="HTTP 404"):
        download_dataset("http://exemplo/data.zip", tmp_path / "d.csv", "d.csv")


@patch("src.data.download.requests.get")
def test_forwards_the_verify_parameter_to_requests(mock_get: MagicMock, tmp_path: Path) -> None:
    """O bundle de CA para redes com inspeção TLS precisa chegar ao requests."""
    mock_get.return_value = MagicMock(content=_fake_zip_bytes("d.csv", b"x"))

    download_dataset(
        "http://exemplo/data.zip", tmp_path / "d.csv", "d.csv", verify="/caminho/ca.pem"
    )

    assert mock_get.call_args.kwargs["verify"] == "/caminho/ca.pem"
