import os
import gc
import csv
import cv2
import pickle
from tqdm import tqdm
import numpy as np
from src.core.pipeline import ExperimentalPipeline
from src.metrics.evaluator import PerformanceEvaluator
from src.utils.model_downloader import ensure_all_onnx_models
from src.core.detector import FaceDetectorTracker

MODELOS = [
    "facenet", 
    "arcface", 
    "cosface", 
    "sphereface", 
    "magface", 
    "curricularface", 
    "elasticface"
]

def extract_detections_from_videos(recordings_dir):
    """Extrai todas as faces e as carrega/salva do cache em disco para evitar reprocessamento."""
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(recordings_dir) if f.endswith(".mp4")]
    if not videos:
        print("  [AVISO] Nenhum vídeo .mp4 encontrado em data/recordings.")
        return {}
        
    cache_dir = os.path.join(os.path.dirname(recordings_dir), "cache")
    os.makedirs(cache_dir, exist_ok=True)
        
    print("\n=== PRÉ-PROCESSAMENTO: Detecção e Rastreamento (YOLO + DeepSort) ===")
    
    all_video_detections = {}
    detector = None
    
    for video in videos:
        cache_path = os.path.join(cache_dir, f"{video}_detections.pkl")
        
        if os.path.exists(cache_path):
            print(f"  -> Carregando detecções do cache em disco para {video}...")
            try:
                with open(cache_path, "rb") as f:
                    detections = pickle.load(f)
                all_video_detections[video] = detections
                print(f"     Carregados {len(detections)} recortes faciais válidos de {video} do cache.")
                continue
            except Exception as e:
                print(f"  [AVISO] Falha ao carregar cache de {video}: {e}. Reprocessando...")
        
        if detector is None:
            # Inicializa sob demanda
            detector = FaceDetectorTracker()
            
        video_path = os.path.join(recordings_dir, video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [AVISO] Falha ao abrir: {video}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc=f"  -> Detectando rostos em {video}",
            unit="f",
            leave=False
        )
        
        detections = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            tracks = detector.process_frame(frame)
            
            for track in tracks:
                track_id = track["track_id"]
                bbox = track["bbox"]  # (xmin, ymin, xmax, ymax)
                
                h_img, w_img = frame.shape[:2]
                xmin = max(0, int(bbox[0]))
                ymin = max(0, int(bbox[1]))
                xmax = min(w_img, int(bbox[2]))
                ymax = min(h_img, int(bbox[3]))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size == 0:
                    continue
                    
                detections.append({
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "face_crop": face_crop.copy() # Importante copiar para que o frame original seja liberado
                })
            
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        all_video_detections[video] = detections
        print(f"     Extraídos {len(detections)} recortes faciais válidos de {video}.")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(detections, f)
        except Exception as e:
            print(f"  [AVISO] Nao foi possivel salvar cache de {video}: {e}")
        
    # Limpa detector da RAM/VRAM para que os modelos tenham memória de sobra
    if detector is not None:
        del detector
        gc.collect()
    
    return all_video_detections


def main():
    print("=== INICIANDO PIPELINE EXPERIMENTAL ERBASE 2026 ===")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(root_dir, "data", "recordings")
    results_dir = os.path.join(root_dir, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    models_dir = os.path.join(root_dir, "data", "models")
    ensure_all_onnx_models(models_dir)
    
    # Etapa 1: Pré-processamento
    all_video_detections = extract_detections_from_videos(recordings_dir)
    if not all_video_detections:
        print("  [AVISO] Nenhuma detecção extraída. Encerrando pipeline.")
        return
        
    evaluator = PerformanceEvaluator()
    
    # Prepara o arquivo summary data.csv por vídeo
    for video in all_video_detections.keys():
        video_result_dir = os.path.join(results_dir, video)
        os.makedirs(video_result_dir, exist_ok=True)
        
        summary_csv_path = os.path.join(video_result_dir, "data.csv")
        if os.path.exists(summary_csv_path):
            try:
                os.remove(summary_csv_path)
            except Exception:
                pass
                
        with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", 
                "accuracy", 
                "auc", 
                "tar_at_far_1%", 
                "recall_at_1", 
                "recall_at_3", 
                "precision_at_1", 
                "precision_at_3", 
                "avg_inference_time_ms"
            ])
            
    # Etapa 2: Avaliação de cada modelo
    for model_name in MODELOS:
        print(f"\n>>> Executando avaliações para o modelo: {model_name.upper()} <<<")
        try:
            # Inicializa o pipeline e o DB
            pipeline = ExperimentalPipeline(model_name=model_name, threshold=0.5)
            
            for video, detections in all_video_detections.items():
                print(f"  -> Processando vídeo: {video}")
                
                results = pipeline.run(video_name=video, detections=detections)
                
                if results["status"] == "success":
                    print(f"     Status: Sucesso | Frames avaliados: {results['frames_processed']}")
                    
                    video_gt = pipeline.ground_truth.get(video, {})
                    track_winners = results["track_winners"]
                    
                    track_results = []
                    for track_id, predicted in track_winners.items():
                        ground_truth = video_gt.get(str(track_id))
                        if ground_truth is not None:
                            is_correct = (predicted == ground_truth)
                            track_results.append({
                                "video": video,
                                "track_id": track_id,
                                "ground_truth": ground_truth,
                                "predicted": predicted,
                                "correct": is_correct
                            })
                            
                    # Métricas do vídeo para este modelo
                    if track_results:
                        y_true_tracks = [t["ground_truth"] for t in track_results]
                        y_pred_tracks = [t["predicted"] for t in track_results]
                        accuracy = evaluator.calculate_accuracy(y_true_tracks, y_pred_tracks)
                    else:
                        accuracy = 0.0
                        
                    frame_logs = results["frame_logs"]
                    if frame_logs:
                        y_true_frames = [f["label"] for f in frame_logs]
                        y_scores_frames = [f["similarity_score"] for f in frame_logs]
                        
                        auc_val = evaluator.calculate_auc(y_true_frames, y_scores_frames)
                        tar_at_far = evaluator.calculate_tar_at_far(y_true_frames, y_scores_frames, far_threshold=0.01)
                        
                        recalls_1 = [evaluator.calculate_recall_at_k(f["true_id"], f["retrieved_ids"], k=1) for f in frame_logs]
                        recalls_3 = [evaluator.calculate_recall_at_k(f["true_id"], f["retrieved_ids"], k=3) for f in frame_logs]
                        precisions_1 = [evaluator.calculate_precision_at_k(f["true_id"], f["retrieved_ids"], k=1) for f in frame_logs]
                        precisions_3 = [evaluator.calculate_precision_at_k(f["true_id"], f["retrieved_ids"], k=3) for f in frame_logs]
                        
                        recall_at_1 = float(np.mean(recalls_1))
                        recall_at_3 = float(np.mean(recalls_3))
                        precision_at_1 = float(np.mean(precisions_1))
                        precision_at_3 = float(np.mean(precisions_3))
                    else:
                        auc_val = 0.0
                        tar_at_far = 0.0
                        recall_at_1 = 0.0
                        recall_at_3 = 0.0
                        precision_at_1 = 0.0
                        precision_at_3 = 0.0
                        
                    avg_time = results["avg_inference_time_ms"]
                    
                    # Salva os track_results (track-level) em data/results/[video]/[model].csv
                    video_result_dir = os.path.join(results_dir, video)
                    model_csv_path = os.path.join(video_result_dir, f"{model_name}.csv")
                    with open(model_csv_path, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["video", "track_id", "ground_truth", "predicted", "correct"])
                        for row in track_results:
                            writer.writerow([
                                row["video"], 
                                row["track_id"], 
                                row["ground_truth"], 
                                row["predicted"], 
                                str(row["correct"])
                            ])
                    print(f"     -> Salvo resultados de tracks em: {model_csv_path}")
                    
                    # Acumula no data.csv do vídeo
                    summary_csv_path = os.path.join(video_result_dir, "data.csv")
                    with open(summary_csv_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            model_name,
                            f"{accuracy:.4f}",
                            f"{auc_val:.4f}",
                            f"{tar_at_far:.4f}",
                            f"{recall_at_1:.4f}",
                            f"{recall_at_3:.4f}",
                            f"{precision_at_1:.4f}",
                            f"{precision_at_3:.4f}",
                            f"{avg_time:.2f}"
                        ])
                    
                else:
                    print(f"     Status: Falha | Motivo: {results.get('reason')}")

            pipeline.cleanup()
            del pipeline
            gc.collect()
            print(f">>> Concluído modelo {model_name.upper()}. Recursos limpos e memória desalocada.\n")
            
        except Exception as e:
            print(f"  [ERRO] Falha crítica na execução do modelo {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("=== PIPELINE EXPERIMENTAL CONCLUÍDO COM SUCESSO ===")

if __name__ == "__main__":
    main()
