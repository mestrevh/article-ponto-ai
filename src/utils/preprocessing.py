import cv2
import numpy as np
from typing import Tuple

class ImagePreprocessor:
    def apply(self, img_bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Aplica CLAHE para normalizar iluminação e redimensiona mantendo os canais em BGR/uint8."""
        if img_bgr is None or img_bgr.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
            
        # Converte para LAB para aplicar CLAHE no canal de luminância L
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Redimensiona para o target_size solicitado
        resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_CUBIC)
        return resized
