import os
from src.core.base import FaceRecognizer
from src.models.deepface_models import FaceNetWrapper, ArcFaceWrapper
from src.models.onnx_models import ONNXFaceRecognizer
from src.utils.helpers import get_project_root
from src.utils.model_downloader import ensure_model

class ModelFactory:
    @staticmethod
    def create(model_name: str) -> FaceRecognizer:
        """Cria e retorna a instância correta do wrapper do modelo."""
        model_lower = model_name.lower()
        if model_lower == "facenet":
            return FaceNetWrapper()
        elif model_lower == "arcface":
            return ArcFaceWrapper()
        elif model_lower in ["cosface", "sphereface", "magface", "curricularface", "elasticface"]:
            # Obtém a raiz do projeto e resolve o caminho do checkpoint .onnx em data/models/
            root = get_project_root()
            models_dir = os.path.join(root, "data", "models")
            model_filename = f"{model_lower}.onnx"
            # Garante que o modelo existe localmente, baixando-o se necessário
            model_path = ensure_model(model_filename, models_dir)
            return ONNXFaceRecognizer(model_name=model_lower, model_path=model_path, dimension=512)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")
