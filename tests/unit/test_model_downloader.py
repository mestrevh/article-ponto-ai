import os
import io
import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.utils.model_downloader import (
    ensure_model,
    ensure_all_onnx_models,
    ONNX_MODEL_URLS,
    _download_file,
)


def test_ensure_model_returns_path_when_file_exists(tmp_path):
    """Se o modelo já existir, retorna o caminho sem tentar download."""
    model_file = tmp_path / "cosface.onnx"
    model_file.write_bytes(b"fake_content")

    result = ensure_model("cosface.onnx", str(tmp_path))

    assert result == str(model_file)


def test_ensure_model_raises_for_unknown_model(tmp_path):
    """Levanta FileNotFoundError para modelo sem URL configurada."""
    with pytest.raises(FileNotFoundError, match="não possui URLs de download configuradas"):
        ensure_model("unknown_model.onnx", str(tmp_path))


@patch("src.utils.model_downloader._download_file", return_value=True)
def test_ensure_model_downloads_when_missing(mock_dl, tmp_path):
    """Chama _download_file quando o modelo está ausente."""
    result = ensure_model("cosface.onnx", str(tmp_path))
    assert mock_dl.called
    assert result.endswith("cosface.onnx")


@patch("src.utils.model_downloader._download_file", return_value=False)
def test_ensure_model_raises_when_all_downloads_fail(mock_dl, tmp_path):
    """Levanta FileNotFoundError quando todas as URLs falham."""
    with pytest.raises(FileNotFoundError):
        ensure_model("cosface.onnx", str(tmp_path))


def test_ensure_all_onnx_models_skips_when_all_present(tmp_path, capsys):
    """Não realiza downloads quando todos os modelos já existem."""
    for name in ONNX_MODEL_URLS:
        (tmp_path / name).write_bytes(b"fake")

    ensure_all_onnx_models(str(tmp_path))

    captured = capsys.readouterr()
    assert "já estão presentes" in captured.out


@patch("src.utils.model_downloader.ensure_model")
def test_ensure_all_onnx_models_downloads_missing(mock_ensure, tmp_path):
    """Chama ensure_model para cada modelo faltante."""
    mock_ensure.return_value = str(tmp_path / "cosface.onnx")

    ensure_all_onnx_models(str(tmp_path))

    assert mock_ensure.call_count == len(ONNX_MODEL_URLS)


@patch("src.utils.model_downloader.ensure_model", side_effect=FileNotFoundError("falha"))
def test_ensure_all_onnx_models_raises_on_failure(mock_ensure, tmp_path):
    """Levanta RuntimeError se algum modelo não puder ser baixado."""
    with pytest.raises(RuntimeError, match="Falha ao obter"):
        ensure_all_onnx_models(str(tmp_path))


def test_download_file_returns_false_on_network_error(tmp_path):
    """_download_file retorna False quando a URL não é acessível."""
    dest = str(tmp_path / "test.onnx")
    result = _download_file("http://invalid.invalid/test.onnx", dest)
    assert result is False
    assert not os.path.exists(dest)


def test_download_file_succeeds_and_returns_true(tmp_path):
    """_download_file retorna True e grava o arquivo quando o download é bem-sucedido."""
    dest = str(tmp_path / "model.onnx")
    fake_data = b"fake_onnx_data"

    mock_response = MagicMock()
    mock_response.headers.get.return_value = str(len(fake_data))
    mock_response.read.side_effect = [fake_data, b""]
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("src.utils.model_downloader.urllib.request.urlopen", return_value=mock_response):
        result = _download_file("https://fake.url/model.onnx", dest)

    assert result is True
    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == fake_data


def test_download_file_cleans_partial_file_on_error(tmp_path):
    """_download_file remove arquivo parcial quando ocorre erro após abertura."""
    dest = str(tmp_path / "model.onnx")
    # Cria um arquivo parcial para simular download interrompido
    with open(dest, "wb") as f:
        f.write(b"partial")

    # Simula erro durante urlopen
    import urllib.error
    with patch(
        "src.utils.model_downloader.urllib.request.urlopen",
        side_effect=urllib.error.URLError("conexão recusada"),
    ):
        result = _download_file("https://fake.url/model.onnx", dest)

    assert result is False
    assert not os.path.exists(dest)


def test_onnx_model_urls_has_all_expected_models():
    """Verifica que todos os 5 modelos ONNX do projeto têm URLs configuradas."""
    expected = {"cosface.onnx", "sphereface.onnx", "magface.onnx", "curricularface.onnx", "elasticface.onnx"}
    assert set(ONNX_MODEL_URLS.keys()) == expected
