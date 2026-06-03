import os
import pytest
from unittest.mock import patch, MagicMock
from src.utils.model_downloader import (
    ensure_model,
    ensure_all_onnx_models,
    ONNX_MODEL_NAMES,
    DOWNLOAD_URLS,
    _download_file,
    _find_existing_model,
    _download_base_model,
)


def test_ensure_model_returns_path_when_file_exists(tmp_path):
    """Se o modelo já existir, retorna o caminho sem tentar download."""
    model_file = tmp_path / "cosface.onnx"
    model_file.write_bytes(b"fake_content")

    result = ensure_model("cosface.onnx", str(tmp_path))

    assert result == str(model_file)


def test_ensure_model_raises_for_unknown_model(tmp_path):
    """Levanta FileNotFoundError para modelo sem nome configurado."""
    with pytest.raises(FileNotFoundError, match="não possui URLs de download configuradas"):
        ensure_model("unknown_model.onnx", str(tmp_path))


def test_ensure_model_copies_from_existing(tmp_path):
    """Se outro modelo existir, copia-o em vez de baixar."""
    existing = tmp_path / "cosface.onnx"
    existing.write_bytes(b"fake_model_data")

    result = ensure_model("magface.onnx", str(tmp_path))
    assert result.endswith("magface.onnx")
    assert (tmp_path / "magface.onnx").exists()
    assert (tmp_path / "magface.onnx").read_bytes() == b"fake_model_data"


@patch("src.utils.model_downloader._download_base_model", return_value=True)
def test_ensure_model_downloads_when_none_exist(mock_dl, tmp_path):
    """Faz download se nenhum modelo existir."""
    result = ensure_model("cosface.onnx", str(tmp_path))
    assert mock_dl.called
    assert result.endswith("cosface.onnx")


@patch("src.utils.model_downloader._download_base_model", return_value=False)
def test_ensure_model_raises_when_download_fails(mock_dl, tmp_path):
    """Levanta FileNotFoundError quando download falha."""
    with pytest.raises(FileNotFoundError, match="Não foi possível baixar"):
        ensure_model("cosface.onnx", str(tmp_path))


def test_ensure_all_skips_when_all_present(tmp_path, capsys):
    """Não faz nada quando todos os modelos existem."""
    for name in ONNX_MODEL_NAMES:
        (tmp_path / name).write_bytes(b"fake")

    ensure_all_onnx_models(str(tmp_path))

    captured = capsys.readouterr()
    assert "já estão presentes" in captured.out


def test_ensure_all_copies_from_existing(tmp_path):
    """Se um modelo existe, copia para os faltantes sem download."""
    (tmp_path / "cosface.onnx").write_bytes(b"model_data")

    ensure_all_onnx_models(str(tmp_path))

    for name in ONNX_MODEL_NAMES:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).read_bytes() == b"model_data"


@patch("src.utils.model_downloader._download_base_model", return_value=True)
def test_ensure_all_downloads_once_and_copies(mock_dl, tmp_path):
    """Quando nenhum modelo existe, baixa uma vez e copia para os demais."""
    # Simula que o download cria o arquivo
    def fake_download(dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"downloaded_model")
        return True
    mock_dl.side_effect = fake_download

    ensure_all_onnx_models(str(tmp_path))

    # Download chamado exatamente 1 vez
    assert mock_dl.call_count == 1

    # Todos os 5 modelos existem
    for name in ONNX_MODEL_NAMES:
        assert (tmp_path / name).exists()


@patch("src.utils.model_downloader._download_base_model", return_value=False)
def test_ensure_all_raises_when_download_fails(mock_dl, tmp_path):
    """Levanta RuntimeError se nenhum modelo existir e download falhar."""
    with pytest.raises(RuntimeError, match="Não foi possível baixar"):
        ensure_all_onnx_models(str(tmp_path))


def test_find_existing_model_returns_first_found(tmp_path):
    """_find_existing_model retorna o primeiro modelo encontrado."""
    (tmp_path / "sphereface.onnx").write_bytes(b"data")
    result = _find_existing_model(str(tmp_path))
    assert result is not None
    assert result.endswith("sphereface.onnx")


def test_find_existing_model_returns_none_when_empty(tmp_path):
    """_find_existing_model retorna None quando nenhum modelo existe."""
    result = _find_existing_model(str(tmp_path))
    assert result is None


def test_find_existing_model_ignores_empty_files(tmp_path):
    """_find_existing_model ignora arquivos vazios (0 bytes)."""
    (tmp_path / "cosface.onnx").write_bytes(b"")
    result = _find_existing_model(str(tmp_path))
    assert result is None


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
    """_download_file remove arquivo parcial quando ocorre erro."""
    dest = str(tmp_path / "model.onnx")
    with open(dest, "wb") as f:
        f.write(b"partial")

    import urllib.error
    with patch(
        "src.utils.model_downloader.urllib.request.urlopen",
        side_effect=urllib.error.URLError("conexão recusada"),
    ):
        result = _download_file("https://fake.url/model.onnx", dest)

    assert result is False
    assert not os.path.exists(dest)


def test_download_base_model_tries_urls_in_order(tmp_path):
    """_download_base_model tenta as URLs em ordem até ter sucesso."""
    dest = str(tmp_path / "model.onnx")

    with patch("src.utils.model_downloader._download_file") as mock_dl:
        mock_dl.side_effect = [False, True]  # primeira falha, segunda sucede
        result = _download_base_model(dest)

    assert result is True
    assert mock_dl.call_count == 2


def test_onnx_model_names_has_all_expected():
    """Verifica que todos os 5 modelos ONNX esperados estão registrados."""
    expected = {"cosface.onnx", "sphereface.onnx", "magface.onnx", "curricularface.onnx", "elasticface.onnx"}
    assert set(ONNX_MODEL_NAMES) == expected
