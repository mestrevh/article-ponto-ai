from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class FaceRecognizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna o nome identificador do modelo (ex: 'arcface')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Retorna a dimensão do vetor de embedding gerado (ex: 512)."""
        pass

    @abstractmethod
    def extract_embedding_with_filters(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Recebe um recorte facial BGR, aplica o pré-processamento uniforme,
        extrai o vetor de características e o retorna L2-normalizado.
        Retorna None se a extração falhar.
        """
        pass

    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """
        Retorna metadados do modelo para tabelas de comparação,
        contendo: {'name': str, 'dim': int, 'init_ms': float, 'provider': str}.
        """
        pass
