import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.deepface_models import FaceNetWrapper, ArcFaceWrapper
from src.models.onnx_models import ONNXFaceRecognizer

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

@patch("src.models.onnx_models.ort.InferenceSession")
@patch("src.models.onnx_models.os.path.exists")
def test_onnx_model_output_is_l2_normalized_and_has_correct_dimension(mock_exists, mock_session):
    """Valida se o embedding do wrapper ONNX tem tamanho 512 e norma unitária."""
    mock_exists.return_value = True
    mock_sess_inst = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_sess_inst.get_inputs.return_value = [mock_input]
    mock_sess_inst.run.return_value = [np.array([[2.0] * 512])]
    mock_session.return_value = mock_sess_inst
    
    recognizer = ONNXFaceRecognizer("cosface", "dummy_path.onnx")
    embedding = recognizer.extract_embedding_with_filters(np.zeros((112, 112, 3), dtype=np.uint8))
    
    assert embedding.shape == (512,)
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
    
    p = recognizer.params()
    assert p["name"] == "cosface"
    assert p["dim"] == 512
    
@patch("src.models.onnx_models.os.path.exists")
def test_onnx_model_raises_file_not_found(mock_exists):
    """Garante erro de arquivo se o modelo ONNX não existir."""
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        ONNXFaceRecognizer("cosface", "non_existent.onnx")
