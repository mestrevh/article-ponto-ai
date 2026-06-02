import os
import cv2
import onnxruntime as ort
import numpy as np
from src.core.base import FaceRecognizer
from typing import Dict, Any, Optional

class ONNXFaceRecognizer(FaceRecognizer):
    def __init__(self, model_name: str, model_path: str, dimension: int = 512):
        self._name = model_name
        self._dimension = dimension
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
            
        # Carrega a sessão de inferência do ONNXRuntime
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def extract_embedding_with_filters(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Recebe um recorte facial BGR, aplica o pré-processamento padrão para
        modelos baseados em iResNet-100, executa inferência e normaliza L2 o embedding.
        """
        try:
            if img_bgr is None or img_bgr.size == 0:
                return None
                
            # Garante tipo uint8 para evitar erros de conversão de cor no OpenCV
            if img_bgr.dtype != np.uint8:
                img_bgr = img_bgr.astype(np.uint8)
                
            # 1. Redimensionamento padrão para iResNet-100 (112x112)
            img_resized = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            # 2. Conversão de BGR para RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 3. Normalização iResNet-100: (x - 127.5) / 128.0
            input_data = (img_rgb.astype(np.float32) - 127.5) / 128.0
            
            # 4. Transposição HWC -> CHW e adição da dimensão de batch (NCHW)
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # 5. Execução da inferência no ONNXRuntime
            outputs = self.session.run(None, {self.input_name: input_data})
            embedding = outputs[0][0]
            
            # 6. Normalização L2
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception:
            return None

    def params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dimension,
            "init_ms": 0.0,
            "provider": "onnx"
        }
