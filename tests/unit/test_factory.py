import pytest
from unittest.mock import patch
from src.models.factory import ModelFactory
from src.models.deepface_models import FaceNetWrapper, ArcFaceWrapper

def test_factory_instantiates_correct_model():
    """Garante a instanciação do wrapper correto para strings válidas."""
    model_facenet = ModelFactory.create("facenet")
    model_arcface = ModelFactory.create("arcface")
    
    assert isinstance(model_facenet, FaceNetWrapper)
    assert isinstance(model_arcface, ArcFaceWrapper)

def test_factory_raises_error_for_invalid_model_string():
    """Garante erro ao solicitar um modelo não suportado."""
    with pytest.raises(ValueError):
        ModelFactory.create("modelo_inexistente")

@patch("src.models.factory.ONNXFaceRecognizer")
def test_factory_instantiates_onnx_models(mock_onnx):
    """Garante a instanciação do wrapper ONNX para strings correspondentes."""
    # Testa um dos modelos ONNX
    model_cosface = ModelFactory.create("cosface")
    assert model_cosface is not None
    mock_onnx.assert_called_once()
