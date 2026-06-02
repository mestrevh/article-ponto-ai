import os
import gc
from src.core.pipeline import ExperimentalPipeline

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
    
    # Caminho do diretório de vídeos
    root_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(root_dir, "data", "recordings")
    
    for model_name in MODELOS:
        print(f"\n>>> Executando avaliações para o modelo: {model_name.upper()} <<<")
        try:
            # Inicializa o pipeline (e realiza o populate isolado por modelo no ChromaDB)
            pipeline = ExperimentalPipeline(model_name=model_name, threshold=0.5)
            
            for video in VIDEOS:
                video_path = os.path.join(recordings_dir, video)
                if not os.path.exists(video_path):
                    print(f"  [AVISO] Vídeo não encontrado: {video_path}")
                    continue
                    
                print(f"  -> Processando vídeo: {video}")
                results = pipeline.run(video_path=video_path)
                
                if results["status"] == "success":
                    print(f"     Status: Sucesso | Frames: {results['frames_processed']}")
                    print(f"     Ganhadores por Track: {results['track_winners']}")
                else:
                    print(f"     Status: Falha | Motivo: {results.get('reason')}")
            
            # Limpeza rígida de memória e ChromaDB
            pipeline.cleanup()
            del pipeline
            gc.collect()
            print(f">>> Concluído modelo {model_name.upper()}. Coleções ChromaDB limpas e memória desalocada.")
            
        except Exception as e:
            print(f"  [ERRO] Falha crítica na execução do modelo {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
