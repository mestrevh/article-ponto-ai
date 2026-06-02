from src.core.base import FaceRecognizer
from src.models.deepface_models import FaceNetWrapper, ArcFaceWrapper

class ModelFactory:
    @staticmethod
    def create(model_name: str) -> FaceRecognizer:
        """Cria e retorna a instância correta do wrapper do modelo."""
        model_lower = model_name.lower()
        if model_lower == "facenet":
            return FaceNetWrapper()
        elif model_lower == "arcface":
            return ArcFaceWrapper()
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")
