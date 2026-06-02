import pytest
import numpy as np
from unittest.mock import patch
from src.models.deepface_models import FaceNetWrapper, ArcFaceWrapper

@patch("src.models.deepface_models.DeepFace.represent")
def test_facenet_output_is_l2_normalized_and_has_correct_dimension(mock_represent):
    """Valida se o embedding gerado tem tamanho 512 e norma unitária (norm = 1.0)."""
    mock_represent.return_value = [{"embedding": [2.0] * 512}]
    
    wrapper = FaceNetWrapper()
    embedding = wrapper.extract_embedding_with_filters(np.zeros((112, 112, 3)))
    
    assert embedding.shape == (512,)
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
    
    p = wrapper.params()
    assert p["name"] == "facenet"
    assert p["dim"] == 512

@patch("src.models.deepface_models.DeepFace.represent")
def test_arcface_output_is_l2_normalized_and_has_correct_dimension(mock_represent):
    """Valida se o embedding gerado por ArcFace tem tamanho 512 e norma unitária (norm = 1.0)."""
    mock_represent.return_value = [{"embedding": [3.0] * 512}]
    
    wrapper = ArcFaceWrapper()
    embedding = wrapper.extract_embedding_with_filters(np.zeros((112, 112, 3)))
    
    assert embedding.shape == (512,)
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
    
    p = wrapper.params()
    assert p["name"] == "arcface"
    assert p["dim"] == 512

