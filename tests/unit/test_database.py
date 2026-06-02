import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.database import VectorDatabase

@patch("src.core.database.chromadb.Client")
def test_database_initialization_creates_isolated_collection(mock_client):
    """Garante que a inicialização do DB cria uma coleção única por modelo."""
    mock_collection = MagicMock()
    mock_client.return_value.get_or_create_collection.return_value = mock_collection
    
    db = VectorDatabase(model_name="arcface")
    
    mock_client.return_value.get_or_create_collection.assert_called_once_with(name="arcface")
    assert db.collection == mock_collection

def test_database_populate_inserts_embeddings():
    """Verifica se o método populate insere corretamente embeddings no banco vetorial."""
    db = VectorDatabase(model_name="facenet")
    db.collection = MagicMock()
    
    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = [0.1] * 512
    
    with patch("src.core.database.os.listdir", return_value=["user1_01.jpg"]), \
         patch("src.core.database.cv2.imread", return_value=np.zeros((100, 100, 3))):
        db.populate(recognizer=mock_recognizer, faces_dir="/dummy/dir")
        
    db.collection.add.assert_called_once()

def test_database_query_returns_correct_match():
    """Garante que a busca retorna o ID da face se estiver dentro do threshold."""
    db = VectorDatabase(model_name="facenet")
    db.collection = MagicMock()
    db.collection.query.return_value = {
        "ids": [["user1"]],
        "distances": [[0.15]]
    }
    
    result = db.query_embedding(embedding=[0.1]*512, threshold=0.4)
    assert result == "user1"

def test_database_query_returns_none_above_threshold():
    """Garante que a busca retorna None se a similaridade cosseno for muito baixa."""
    db = VectorDatabase(model_name="facenet")
    db.collection = MagicMock()
    db.collection.query.return_value = {
        "ids": [["user1"]],
        "distances": [[0.6]]
    }
    
    result = db.query_embedding(embedding=[0.1]*512, threshold=0.4)
    assert result is None
