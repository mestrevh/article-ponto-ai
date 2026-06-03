import pytest
from unittest.mock import patch, MagicMock
import gc
import numpy as np
from src.core.pipeline import ExperimentalPipeline

@patch("src.core.pipeline.VectorDatabase")
@patch("src.core.pipeline.ModelFactory")
def test_pipeline_execution_and_resource_cleanup(mock_factory, mock_db_class):
    """Garante a execução completa do fluxo e a liberação correta de memória RAM/VRAM."""
    # Mock DB com query_top_k retornando estrutura de dicionário válida
    mock_db = MagicMock()
    mock_db.query_top_k.return_value = {
        "ids": [["pessoa_1", "pessoa_2"]],
        "distances": [[0.1, 0.4]]
    }
    mock_db_class.return_value = mock_db
    
    # Mock recognizer
    mock_recognizer = MagicMock()
    mock_recognizer.extract_embedding_with_filters.return_value = np.zeros(512)
    mock_factory.create.return_value = mock_recognizer
    
    # Prepara deteções mockadas
    mock_detections = [
        {"frame_idx": 1, "track_id": 1, "face_crop": np.zeros((112, 112, 3), dtype=np.uint8)},
        {"frame_idx": 2, "track_id": 1, "face_crop": np.zeros((112, 112, 3), dtype=np.uint8)}
    ]
    
    # Adiciona mock do ground_truth ao pipeline
    pipeline = ExperimentalPipeline(model_name="facenet")
    # Define ground truth para o track_id 1
    pipeline.ground_truth = {
        "dummy_video.mp4": {
            "1": "pessoa_1"
        }
    }
    
    results = pipeline.run(video_name="dummy_video.mp4", detections=mock_detections)
    
    assert results is not None
    assert results["status"] == "success"
    assert results["frames_processed"] == 2
    assert results["track_winners"][1] == "pessoa_1"
    
    pipeline.cleanup()
    del pipeline
    collected = gc.collect()
    assert collected >= 0
