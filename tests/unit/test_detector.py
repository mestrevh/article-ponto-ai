import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.detector import FaceDetectorTracker

@patch("src.core.detector.YOLO")
@patch("src.core.detector.DeepSort")
def test_detector_returns_empty_when_no_detections(mock_deepsort, mock_yolo):
    """Garante retorno vazio se o YOLO não detectar rostos."""
    mock_yolo.return_value.predict.return_value = []
    
    detector = FaceDetectorTracker()
    tracks = detector.process_frame(frame=np.zeros((480, 640, 3)))
    
    assert tracks == []

@patch("src.core.detector.YOLO")
@patch("src.core.detector.DeepSort")
def test_detector_returns_tracks_with_ids(mock_deepsort, mock_yolo):
    """Garante que detecções válidas são mapeadas em objetos de rastreamento com ID."""
    mock_yolo.return_value.predict.return_value = [MagicMock()]
    mock_deepsort.return_value.update.return_value = [
        [100, 100, 200, 200, 1]
    ]
    
    detector = FaceDetectorTracker()
    tracks = detector.process_frame(frame=np.zeros((480, 640, 3)))
    
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["bbox"] == (100, 100, 200, 200)
