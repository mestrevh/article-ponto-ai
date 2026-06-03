"""
Testes adicionais para src/core/database.py cobrindo os ramos não testados.
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.database import VectorDatabase


def test_database_populate_real_directory(tmp_path):
    """Cobre o ramo os.walk quando faces_dir existe de verdade."""
    # Cria estrutura de diretórios com subpastas
    person_dir = tmp_path / "pessoa_1"
    person_dir.mkdir()
    img_path = person_dir / "foto.jpg"
    img_path.write_bytes(b"fake_jpg")

    db = VectorDatabase(model_name="test_walk")
    db.collection = MagicMock()
    db.collection.count.return_value = 0

    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = [0.1] * 512

    with patch("src.core.database.cv2.imread", return_value=np.zeros((112, 112, 3))):
        db.populate(recognizer=mock_recognizer, faces_dir=str(tmp_path))

    db.collection.add.assert_called()


def test_database_populate_flat_directory(tmp_path):
    """Cobre o ramo sem subpastas (face_id via partição de '_')."""
    img_path = tmp_path / "user1_01.jpg"
    img_path.write_bytes(b"fake")

    db = VectorDatabase(model_name="test_flat")
    db.collection = MagicMock()
    db.collection.count.return_value = 0

    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = [0.5] * 512

    with patch("src.core.database.cv2.imread", return_value=np.zeros((112, 112, 3))):
        db.populate(recognizer=mock_recognizer, faces_dir=str(tmp_path))

    db.collection.add.assert_called_once()


def test_database_populate_skips_when_imread_returns_none(tmp_path):
    """Se cv2.imread retornar None, não insere no banco."""
    img_path = tmp_path / "user1_01.jpg"
    img_path.write_bytes(b"fake")

    db = VectorDatabase(model_name="test_none")
    db.collection = MagicMock()
    db.collection.count.return_value = 0

    mock_recognizer = MagicMock()

    with patch("src.core.database.cv2.imread", return_value=None):
        db.populate(recognizer=mock_recognizer, faces_dir=str(tmp_path))

    db.collection.add.assert_not_called()


def test_database_populate_skips_when_embedding_is_none(tmp_path):
    """Se o recognizer retornar None para o embedding, não insere no banco."""
    img_path = tmp_path / "user1_01.jpg"
    img_path.write_bytes(b"fake")

    db = VectorDatabase(model_name="test_emb_none")
    db.collection = MagicMock()
    db.collection.count.return_value = 0

    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = None

    with patch("src.core.database.cv2.imread", return_value=np.zeros((112, 112, 3))):
        db.populate(recognizer=mock_recognizer, faces_dir=str(tmp_path))

    db.collection.add.assert_not_called()


def test_database_query_embedding_returns_none_on_empty_results():
    """query_embedding retorna None quando a coleção não tem resultados."""
    db = VectorDatabase(model_name="test_empty")
    db.collection = MagicMock()
    db.collection.query.return_value = {"ids": [[]], "distances": [[]]}

    result = db.query_embedding(embedding=[0.1] * 512, threshold=0.5)
    assert result is None


def test_database_query_top_k_returns_results():
    """query_top_k retorna o dicionário de resultados da coleção."""
    db = VectorDatabase(model_name="test_topk")
    db.collection = MagicMock()
    expected = {"ids": [["u1", "u2"]], "distances": [[0.1, 0.3]]}
    db.collection.query.return_value = expected

    result = db.query_top_k(embedding=[0.1] * 512, k=2)
    assert result == expected


def test_database_query_top_k_returns_empty_on_exception():
    """query_top_k retorna {} quando a query lança exceção."""
    db = VectorDatabase(model_name="test_topk_exc")
    db.collection = MagicMock()
    db.collection.query.side_effect = Exception("erro simulado")

    result = db.query_top_k(embedding=[0.1] * 512, k=3)
    assert result == {}


def test_database_cleanup_deletes_collection():
    """cleanup não deve lançar exceções."""
    db = VectorDatabase(model_name="test_cleanup")
    db.client = MagicMock()
    db.cleanup()
    db.client.delete_collection.assert_not_called()


def test_database_cleanup_silences_exceptions():
    """cleanup não propaga exceções."""
    db = VectorDatabase(model_name="test_cleanup_exc")
    db.cleanup()  # Não deve lançar
