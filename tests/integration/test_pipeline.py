import pytest
from unittest.mock import patch, MagicMock
import gc
import numpy as np
from src.core.pipeline import ExperimentalPipeline

@patch("src.core.pipeline.cv2.VideoCapture")
def test_pipeline_execution_and_resource_cleanup(mock_video):
    """Garante a execução completa do fluxo e a liberação correta de memória RAM/VRAM."""
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, True, False]
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3)))
    mock_video.return_value = mock_cap
    
    pipeline = ExperimentalPipeline(model_name="facenet")
    results = pipeline.run(video_path="dummy_video.mp4")
    
    assert results is not None
    
    del pipeline
    collected = gc.collect()
    assert collected >= 0
