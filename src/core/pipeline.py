import cv2
import gc
import os
import numpy as np
from typing import Dict, Any, List, Optional
from src.core.base import FaceRecognizer
from src.models.factory import ModelFactory
from src.core.database import VectorDatabase
from src.core.detector import FaceDetectorTracker
from src.core.voting import TrackVotingSystem
from src.utils.preprocessing import ImagePreprocessor

class ExperimentalPipeline:
    def __init__(self, model_name: str, threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.recognizer = ModelFactory.create(model_name)
        self.db = VectorDatabase(model_name)
        self.detector = FaceDetectorTracker()
        self.preprocessor = ImagePreprocessor()
        
        # Popular o banco vetorial se o diretório de treinamento existir
        training_dir = os.path.join("data", "training_faces")
        if os.path.exists(training_dir):
            self.db.populate(self.recognizer, training_dir)

    def run(self, video_path: str) -> Dict[str, Any]:
        """Executa detecção, tracking, extração, busca e votação para o vídeo."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "model": self.model_name,
                "status": "failed",
                "reason": f"Could not open video file: {video_path}"
            }
            
        voting_system = TrackVotingSystem(min_votes=1)
        frames_processed = 0
        frame_logs = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # 1. Detecção e Tracking
            tracks = self.detector.process_frame(frame)
            
            # 2. Para cada rastro (track) ativo
            for track in tracks:
                track_id = track["track_id"]
                bbox = track["bbox"]  # (xmin, ymin, xmax, ymax)
                
                # Garante os limites da imagem
                h_img, w_img = frame.shape[:2]
                xmin = max(0, int(bbox[0]))
                ymin = max(0, int(bbox[1]))
                xmax = min(w_img, int(bbox[2]))
                ymax = min(h_img, int(bbox[3]))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # Extrai o recorte do rosto
                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size == 0:
                    continue
                    
                # 3. Pré-processamento (Upscale + CLAHE)
                preprocessed_face = self.preprocessor.apply(face_crop, (112, 112))
                
                # 4. Extrair embedding L2-normalizado
                embedding = self.recognizer.extract_embedding_with_filters(preprocessed_face)
                if embedding is None:
                    continue
                
                # 5. Busca no banco vetorial
                matched_id = self.db.query_embedding(embedding, self.threshold)
                identity = matched_id if matched_id is not None else "Desconhecido"
                
                # 6. Acumular voto
                voting_system.add_vote(track_id, identity)
                
                # Calcular score de similaridade (1.0 - distância)
                # Para fins de ROC/AUC, precisamos da distância da query
                try:
                    results = self.db.collection.query(
                        query_embeddings=[list(map(float, embedding))],
                        n_results=1
                    )
                    distance = 1.0
                    if results and results.get("distances") and len(results["distances"][0]) > 0:
                        distance = results["distances"][0][0]
                    similarity = 1.0 - distance
                except Exception:
                    similarity = 0.0
                
                frame_logs.append({
                    "frame": frames_processed,
                    "track_id": track_id,
                    "predicted": identity,
                    "similarity": similarity
                })
                
        cap.release()
        
        # Consolidação de vencedores por track
        track_winners = {}
        for track_id in voting_system.votes.keys():
            winner = voting_system.get_winner(track_id)
            track_winners[track_id] = winner
            
        return {
            "model": self.model_name,
            "frames_processed": frames_processed,
            "track_winners": track_winners,
            "frame_logs": frame_logs,
            "status": "success"
        }

    def cleanup(self) -> None:
        """Limpa conexões e remove coleções para evitar vazamento de memória."""
        if hasattr(self, 'db') and self.db:
            self.db.cleanup()
        # Deletar referências para forçar coleta de lixo
        if hasattr(self, 'recognizer'):
            del self.recognizer
        if hasattr(self, 'detector'):
            del self.detector
        gc.collect()
