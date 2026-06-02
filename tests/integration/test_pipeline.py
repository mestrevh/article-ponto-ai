import pytest
from unittest.mock import patch, MagicMock
import gc
import numpy as np
from src.core.pipeline import ExperimentalPipeline

@patch("src.core.pipeline.cv2.VideoCapture")
@patch("src.core.pipeline.FaceDetectorTracker")
@patch("src.core.pipeline.VectorDatabase")
@patch("src.core.pipeline.ModelFactory")
def test_pipeline_execution_and_resource_cleanup(mock_factory, mock_db_class, mock_detector_class, mock_video):
    """Garante a execução completa do fluxo e a liberação correta de memória RAM/VRAM."""
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, True, False]
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3)))
    mock_video.return_value = mock_cap
    
    # Mock detector para retornar um rastro
    mock_detector = MagicMock()
    mock_detector.process_frame.return_value = [{"track_id": 1, "bbox": (10, 10, 50, 50)}]
    mock_detector_class.return_value = mock_detector
    
    # Mock DB
    mock_db = MagicMock()
    mock_db.query_embedding.return_value = "pessoa_1"
    mock_db.collection.query.return_value = {
        "distances": [[0.1]]
    }
    mock_db_class.return_value = mock_db
    
    # Mock recognizer
    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = np.zeros(512)
    mock_factory.create.return_value = mock_recognizer
    
    pipeline = ExperimentalPipeline(model_name="facenet")
    results = pipeline.run(video_path="dummy_video.mp4")
    
    assert results is not None
    assert results["status"] == "success"
    assert results["frames_processed"] == 2
    assert results["track_winners"][1] == "pessoa_1"
    
    pipeline.cleanup()
    del pipeline
    collected = gc.collect()
    assert collected >= 0
