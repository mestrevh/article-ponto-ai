import cv2
import gc
import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
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
        self.preprocessor = ImagePreprocessor()
        
        # Popular o banco vetorial se o diretório de treinamento existir
        training_dir = os.path.join("data", "training_faces")
        if os.path.exists(training_dir):
            self.db.populate(self.recognizer, training_dir)
            
        # Carrega o Ground Truth de alvos (target.json) se existir
        target_path = os.path.join("data", "target.json")
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                self.ground_truth = json.load(f)
        else:
            self.ground_truth = {}

    def run(self, video_name: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executa extração, busca e votação baseada em recortes faciais pré-computados."""
        voting_system = TrackVotingSystem(min_votes=1)
        frame_logs = []
        inference_times = []
        
        # Mapeamento do Ground Truth específico deste vídeo
        video_gt = self.ground_truth.get(video_name, {})
        
        pbar = tqdm(
            total=len(detections),
            desc=f"  -> Inferindo {video_name}",
            unit="face",
            leave=False
        )
        
        frames_processed = len(detections) # Número de recortes avaliados
        
        for det in detections:
            track_id = det["track_id"]
            face_crop = det["face_crop"]
            frame_idx = det["frame_idx"]
            
            # 3. Pré-processamento (Upscale + CLAHE)
            preprocessed_face = self.preprocessor.apply(face_crop, (112, 112))
            
            # 4. Extrair embedding L2-normalizado
            t_start = time.perf_counter()
            embedding = self.recognizer.extract_embedding_with_filters(preprocessed_face)
            t_end = time.perf_counter()
            if embedding is None:
                pbar.update(1)
                continue
            inference_times.append((t_end - t_start) * 1000.0)
            
            # 5. Busca dos top-5 vizinhos no banco vetorial
            results = self.db.query_top_k(embedding, k=5)
            
            matched_id = None
            similarity_score = 0.0
            retrieved_ids = []
            
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                retrieved_ids = results["ids"][0]
                distances = results["distances"][0]
                
                # Similaridade cosseno = 1.0 - distancia de cosseno
                top_distance = distances[0]
                similarity_score = 1.0 - top_distance
                
                # Aplica threshold de cosseno
                if top_distance <= self.threshold:
                    matched_id = retrieved_ids[0]
            
            identity = matched_id if matched_id is not None else "Desconhecido"
            
            # 6. Acumular voto
            voting_system.add_vote(track_id, identity)
            
            # Determina o Ground Truth correspondente
            true_id = video_gt.get(str(track_id))
            
            # Apenas grava logs frame-a-frame de métricas se houver ground truth mapeado
            if true_id is not None:
                if true_id == "Desconhecido":
                    label = 0
                else:
                    label = 1 if (len(retrieved_ids) > 0 and retrieved_ids[0] == true_id) else 0
                    
                frame_logs.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "true_id": true_id,
                    "predicted_id": identity,
                    "similarity_score": similarity_score,
                    "label": label,
                    "retrieved_ids": retrieved_ids
                })
            pbar.update(1)
            
        pbar.close()
        
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
            "avg_inference_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
            "status": "success"
        }

    def cleanup(self) -> None:
        """Limpa conexões e remove coleções para evitar vazamento de memória."""
        if hasattr(self, 'db') and self.db:
            self.db.cleanup()
        # Deletar referências para forçar coleta de lixo
        if hasattr(self, 'recognizer'):
            del self.recognizer
        gc.collect()
