import pytest
import numpy as np

@pytest.fixture
def sample_image_bgr() -> np.ndarray:
    """Gera uma imagem BGR dummy de 112x112 (tamanho comum de entrada de modelos)."""
    return np.zeros((112, 112, 3), dtype=np.uint8)

@pytest.fixture
def sample_embedding_512() -> np.ndarray:
    """Gera um vetor de embedding de dimensão 512 L2-normalizado."""
    vector = np.random.rand(512).astype(np.float32)
    return vector / np.linalg.norm(vector)

@pytest.fixture
def sample_embedding_128() -> np.ndarray:
    """Gera um vetor de embedding de dimensão 128 L2-normalizado."""
    vector = np.random.rand(128).astype(np.float32)
    return vector / np.linalg.norm(vector)
