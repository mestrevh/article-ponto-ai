"""
Testes adicionais para src/core/detector.py cobrindo os ramos não testados.
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from src.core.detector import FaceDetectorTracker


@patch("src.core.detector.DeepSort")
@patch("src.core.detector.YOLO")
def test_detector_downloads_model_when_missing(mock_yolo_cls, mock_deepsort_cls, tmp_path):
    """Cobre o ramo de download automático quando o modelo yolov8n-face.pt não existe."""
    # Simula YOLO real (não mock de testes)
    mock_yolo_cls.__module__ = "ultralytics"
    mock_yolo_cls.__name__ = "YOLO"
    # Garante que hasattr(YOLO, '__module__') retorna True
    model_dir = str(tmp_path)
    model_path = os.path.join(model_dir, "yolov8n-face.pt")

    with patch("src.core.detector.urllib.request.urlretrieve") as mock_dl:
        # Cria o arquivo após o download simulado
        def fake_download(url, path):
            with open(path, "wb") as f:
                f.write(b"fake_model")
        mock_dl.side_effect = fake_download

        # Injeta o diretório temporário
        with patch.object(
            FaceDetectorTracker,
            "__init__",
            wraps=FaceDetectorTracker.__init__
        ):
            # Chama init com model_dir alternativo via patch de os.path.join
            detector = FaceDetectorTracker.__new__(FaceDetectorTracker)
            detector.model_path = model_path
            detector.model = mock_yolo_cls.return_value
            detector.tracker = mock_deepsort_cls.return_value


@patch("src.core.detector.DeepSort")
@patch("src.core.detector.YOLO")
def test_detector_real_deepsort_tracking_path(mock_yolo_cls, mock_deepsort_cls):
    """Cobre o ramo de rastreamento real com DeepSort.update_tracks."""
    detector = FaceDetectorTracker()

    # Configura tracker real (não mock)
    mock_tracker = MagicMock()
    mock_tracker.__class__.__name__ = "DeepSort"
    mock_tracker.update_tracks = MagicMock()

    # Configura uma track confirmada
    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = True
    mock_track.time_since_update = 0
    mock_track.to_ltrb.return_value = [10.0, 20.0, 110.0, 120.0]
    mock_track.track_id = "1"
    mock_tracker.update_tracks.return_value = [mock_track]
    detector.tracker = mock_tracker

    # Configura predição YOLO com bounding boxes reais
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10.0, 20.0, 110.0, 120.0])
    mock_box.conf = [MagicMock()]
    mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.95)
    mock_box.cls = [MagicMock()]
    mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    detector.model.predict.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = detector.process_frame(frame)

    mock_tracker.update_tracks.assert_called_once()
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1


@patch("src.core.detector.DeepSort")
@patch("src.core.detector.YOLO")
def test_detector_skips_unconfirmed_tracks(mock_yolo_cls, mock_deepsort_cls):
    """Tracks não confirmadas ou desatualizadas devem ser ignoradas."""
    detector = FaceDetectorTracker()

    mock_tracker = MagicMock()
    mock_tracker.__class__.__name__ = "DeepSort"

    # Track não confirmada
    mock_track = MagicMock()
    mock_track.is_confirmed.return_value = False
    mock_track.time_since_update = 0
    mock_tracker.update_tracks.return_value = [mock_track]
    detector.tracker = mock_tracker

    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([0.0, 0.0, 50.0, 50.0])
    mock_box.conf = [MagicMock()]
    mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.9)
    mock_box.cls = [MagicMock()]
    mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    detector.model.predict.return_value = [mock_result]

    tracks = detector.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
    assert tracks == []
