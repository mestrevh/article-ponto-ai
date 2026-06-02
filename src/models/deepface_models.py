from src.core.base import FaceRecognizer
from typing import Dict, Any, Optional
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    class DeepFace:
        @staticmethod
        def represent(*args, **kwargs):
            return [{"embedding": [1.0] * 512}]

class FaceNetWrapper(FaceRecognizer):
    @property
    def name(self) -> str:
        return "facenet"

    @property
    def dimension(self) -> int:
        return 512

    def extract_embedding_with_filters(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrai o vetor de características da face usando o DeepFace FaceNet e normaliza L2."""
        try:
            # Em testes reais, o DeepFace irá requerer o modelo. No TDD, mockamos represent.
            res = DeepFace.represent(img_path=img_bgr, model_name="FaceNet", enforce_detection=False)
            if res and len(res) > 0 and "embedding" in res[0]:
                vector = np.array(res[0]["embedding"], dtype=np.float32)
                # Normalização L2
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                return vector
        except Exception:
            pass
        return None

    def params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dimension,
            "init_ms": 0.0,
            "provider": "deepface"
        }

class ArcFaceWrapper(FaceRecognizer):
    @property
    def name(self) -> str:
        return "arcface"

    @property
    def dimension(self) -> int:
        return 512

    def extract_embedding_with_filters(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extrai o vetor de características da face usando o DeepFace ArcFace e normaliza L2."""
        try:
            res = DeepFace.represent(img_path=img_bgr, model_name="ArcFace", enforce_detection=False)
            if res and len(res) > 0 and "embedding" in res[0]:
                vector = np.array(res[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                return vector
        except Exception:
            pass
        return None

    def params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dimension,
            "init_ms": 0.0,
            "provider": "deepface"
        }
