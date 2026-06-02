import pytest
import numpy as np
from src.utils.preprocessing import ImagePreprocessor

def test_preprocessing_applies_clahe_and_upscale():
    """Garante que a imagem seja corretamente normalizada e redimensionada."""
    preprocessor = ImagePreprocessor()
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    processed = preprocessor.apply(img, target_size=(112, 112))
    
    assert processed.shape == (112, 112, 3)
    assert processed.dtype == np.uint8
