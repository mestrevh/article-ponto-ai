import os
import gc
import csv
import numpy as np
from src.core.pipeline import ExperimentalPipeline
from src.metrics.evaluator import PerformanceEvaluator

MODELOS = [
    "facenet", 
    "arcface", 
    "cosface", 
    "sphereface", 
    "magface", 
    "curricularface", 
    "elasticface"
]

VIDEOS = [
    "camera-externa-video-sozinho.mp4",
    "camera-externa-video-dupla.mp4",
    "camera-externa-video-grupo.mp4",
    "camera-interna-video-sozinho.mp4",
    "camera-interna-video-dupla.mp4",
    "camera-interna-video-grupo.mp4"
]

def main():
    print("=== INICIANDO PIPELINE EXPERIMENTAL ERBASE 2026 ===")
    
    # Caminho do diretório de vídeos e resultados
    root_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(root_dir, "data", "recordings")
    results_dir = os.path.join(root_dir, "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    evaluator = PerformanceEvaluator()
    summary_csv_path = os.path.join(results_dir, "data.csv")
    
    # Prepara o arquivo de resumo comparativo global (data.csv)
    # Se o arquivo já existir, remove para recriar limpo
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
    
    for model_name in MODELOS:
        print(f"\n>>> Executando avaliações para o modelo: {model_name.upper()} <<<")
        try:
            # Inicializa o pipeline
            pipeline = ExperimentalPipeline(model_name=model_name, threshold=0.5)
            
            track_results = []
            all_frame_logs = []
            video_inference_times = []
            
            for video in VIDEOS:
                video_path = os.path.join(recordings_dir, video)
                if not os.path.exists(video_path):
                    print(f"  [AVISO] Vídeo não encontrado: {video_path}")
                    continue
                    
                print(f"  -> Processando vídeo: {video}")
                results = pipeline.run(video_path=video_path)
                
                if results["status"] == "success":
                    print(f"     Status: Sucesso | Frames: {results['frames_processed']}")
                    
                    # 1. Acumula resultados de rastreamento (track_winners)
                    video_gt = pipeline.ground_truth.get(video, {})
                    track_winners = results["track_winners"]
                    for track_id, predicted in track_winners.items():
                        ground_truth = video_gt.get(str(track_id))
                        # Apenas avalia se o track possuir ground truth em target.json
                        if ground_truth is not None:
                            is_correct = (predicted == ground_truth)
                            track_results.append({
                                "video": video,
                                "track_id": track_id,
                                "ground_truth": ground_truth,
                                "predicted": predicted,
                                "correct": is_correct
                            })
                    
                    # 2. Acumula logs frame-a-frame de métricas
                    all_frame_logs.extend(results["frame_logs"])
                    
                    # 3. Guarda tempo médio de inferência do vídeo
                    video_inference_times.append(results["avg_inference_time_ms"])
                else:
                    print(f"     Status: Falha | Motivo: {results.get('reason')}")
            
            # --- Cálculo de Métricas Finais do Modelo ---
            
            # Acurácia (Track-level)
            if track_results:
                y_true_tracks = [t["ground_truth"] for t in track_results]
                y_pred_tracks = [t["predicted"] for t in track_results]
                accuracy = evaluator.calculate_accuracy(y_true_tracks, y_pred_tracks)
            else:
                accuracy = 0.0
                
            # Métricas Frame-level (AUC, TAR@FAR, Precision/Recall@K)
            if all_frame_logs:
                y_true_frames = [f["label"] for f in all_frame_logs]
                y_scores_frames = [f["similarity_score"] for f in all_frame_logs]
                
                # AUC e TAR@FAR 1%
                auc_val = evaluator.calculate_auc(y_true_frames, y_scores_frames)
                tar_at_far = evaluator.calculate_tar_at_far(y_true_frames, y_scores_frames, far_threshold=0.01)
                
                # Recall@K e Precision@K
                recalls_1 = [evaluator.calculate_recall_at_k(f["true_id"], f["retrieved_ids"], k=1) for f in all_frame_logs]
                recalls_3 = [evaluator.calculate_recall_at_k(f["true_id"], f["retrieved_ids"], k=3) for f in all_frame_logs]
                precisions_1 = [evaluator.calculate_precision_at_k(f["true_id"], f["retrieved_ids"], k=1) for f in all_frame_logs]
                precisions_3 = [evaluator.calculate_precision_at_k(f["true_id"], f["retrieved_ids"], k=3) for f in all_frame_logs]
                
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
                
            # Tempo Médio de Extração (ms)
            avg_time = float(np.mean(video_inference_times)) if video_inference_times else 0.0
            
            # --- Gravação dos Arquivos de Saída ---
            
            # 1. Exporta [modelo].csv
            model_csv_path = os.path.join(results_dir, f"{model_name}.csv")
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
            
            # 2. Append no data.csv
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
            print(f"     -> Acumulado resumo comparativo em: {summary_csv_path}")
            
            # Limpeza rígida de memória e coleções
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
