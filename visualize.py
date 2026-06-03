"""
visualize.py — Pipeline de Visualização em Tempo Real

Script independente do main.py e do pipeline de avaliação (src/).
Abre uma janela OpenCV com o vídeo rodando frame a frame, desenhando:
  - Bounding boxes das faces detectadas pelo YOLO
  - Identidade mais provável segundo o ChromaDB
  - Similaridade cosseno (1 - distância)
  - Tamanho do crop (WxH) em cada detecção

Fluxo:
  1. Popula o ChromaDB com embeddings de TODOS os modelos (data/training_faces)
  2. Permite escolher qual modelo usar para a busca vetorial
  3. Permite escolher qual vídeo reproduzir
  4. Reproduz o vídeo com overlay visual de reconhecimento
"""

import os
import sys
import cv2
import time
import numpy as np
import chromadb
from typing import List, Dict, Any, Optional

# ── Imports do projeto ───────────────────────────────────────────────────────
from src.models.factory import ModelFactory
from src.core.base import FaceRecognizer
from src.utils.model_downloader import ensure_all_onnx_models

# ── Constantes ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(ROOT_DIR, "data", "training_faces")
RECORDINGS_DIR = os.path.join(ROOT_DIR, "data", "recordings")
CHROMA_DIR = os.path.join(ROOT_DIR, "data", "chroma_db")
MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")

ALL_MODELS = [
    "facenet",
    "arcface",
    "cosface",
    "sphereface",
    "magface",
    "curricularface",
    "elasticface",
]

# Cores para cada identidade (BGR)
IDENTITY_COLORS = {
    "pessoa_1": (0, 255, 128),    # verde
    "pessoa_2": (255, 128, 0),    # azul-laranja
    "pessoa_3": (0, 200, 255),    # amarelo
    "pessoa_4": (255, 0, 200),    # magenta
    "Desconhecido": (80, 80, 80), # cinza
}
DEFAULT_COLOR = (200, 200, 200)

# ── Configuração do YOLO (import isolado) ────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERRO] ultralytics não instalado. Execute: pip install ultralytics")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def get_color(identity: str) -> tuple:
    """Retorna uma cor BGR consistente para a identidade."""
    return IDENTITY_COLORS.get(identity, DEFAULT_COLOR)


def populate_all_models(training_dir: str, chroma_dir: str) -> None:
    """Popula o ChromaDB com embeddings de TODOS os modelos disponíveis."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   POPULANDO CHROMADB COM TODOS OS MODELOS DE EMBEDDING     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not os.path.exists(training_dir):
        print(f"[ERRO] Diretório de treinamento não encontrado: {training_dir}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=chroma_dir)

    for model_name in ALL_MODELS:
        collection = client.get_or_create_collection(name=model_name)
        if collection.count() > 0:
            print(f"  ✓ {model_name:18s} → já populado ({collection.count()} embeddings)")
            continue

        print(f"  ⏳ {model_name:18s} → extraindo embeddings...", end="", flush=True)
        try:
            recognizer = ModelFactory.create(model_name)
        except Exception as e:
            print(f" ERRO ao criar modelo: {e}")
            continue

        count = 0
        for root, _, filenames in os.walk(training_dir):
            for filename in filenames:
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(root, training_dir)
                face_id = rel_path if rel_path != "." else os.path.splitext(filename)[0].split("_")[0]

                img = cv2.imread(filepath)
                if img is None:
                    continue
                embedding = recognizer.extract_embedding_with_filters(img)
                if embedding is None:
                    continue

                # ID único para evitar duplicatas
                unique_id = f"{face_id}_{filename}"
                collection.add(
                    embeddings=[list(map(float, embedding))],
                    ids=[unique_id],
                    metadatas=[{"filename": filename, "person": face_id}],
                )
                count += 1

        print(f" {count} embeddings salvos")

        # Libera memória do modelo
        del recognizer

    print()


def choose_option(prompt: str, options: List[str]) -> str:
    """Menu interativo no terminal."""
    print(f"\n{'─' * 60}")
    print(f"  {prompt}")
    print(f"{'─' * 60}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print(f"{'─' * 60}")

    while True:
        try:
            choice = int(input("  → Escolha: ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except (ValueError, EOFError):
            pass
        print("  [!] Opção inválida. Tente novamente.")


def draw_overlay(
    frame: np.ndarray,
    bbox: tuple,
    identity: str,
    similarity: float,
    crop_w: int,
    crop_h: int,
    track_id: int,
    conf: float,
) -> None:
    """Desenha o bounding box, identidade, similaridade e tamanho do crop no frame."""
    x1, y1, x2, y2 = map(int, bbox)
    color = get_color(identity)

    # Bounding box principal
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Etiqueta superior: identidade + track
    label_top = f"{identity} (T{track_id})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (tw, th), _ = cv2.getTextSize(label_top, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label_top, (x1 + 4, y1 - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Etiqueta inferior: similaridade + tamanho do crop
    sim_pct = similarity * 100
    label_bot = f"Sim: {sim_pct:.1f}% | Crop: {crop_w}x{crop_h} | Conf: {conf:.0%}"
    (tw2, th2), _ = cv2.getTextSize(label_bot, font, 0.45, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw2 + 8, y2 + th2 + 10), (30, 30, 30), -1)
    cv2.putText(frame, label_bot, (x1 + 4, y2 + th2 + 5), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, model_name: str, video_name: str, frame_idx: int, total_frames: int, fps: float) -> None:
    """Desenha um HUD com informações gerais no topo do frame."""
    h, w = frame.shape[:2]

    # Fundo semi-transparente no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Modelo ativo
    cv2.putText(frame, f"Modelo: {model_name.upper()}", (10, 20), font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

    # Vídeo + frame
    progress = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, progress, (10, 42), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_text, font, 0.55, 1)
    cv2.putText(frame, fps_text, (w - fw - 15, 20), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # Vídeo
    (vw, _), _ = cv2.getTextSize(video_name, font, 0.5, 1)
    cv2.putText(frame, video_name, (w - vw - 15, 42), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Legenda no canto inferior (instruções)
    instructions = "[Q] Sair  |  [P] Pausar  |  [N] Próximo vídeo"
    cv2.putText(frame, instructions, (10, h - 12), font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE DE VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization(model_name: str, video_path: str, video_name: str) -> bool:
    """
    Reproduz o vídeo com detecção YOLO + busca no ChromaDB em tempo real.
    Retorna True se o usuário quer continuar para o próximo, False se quer sair.
    """
    print(f"\n▶ Reproduzindo: {video_name}")
    print(f"  Modelo: {model_name.upper()}")
    print(f"  Controles: [Q] Sair  |  [P] Pausar/Continuar  |  [N] Próximo vídeo\n")

    # ── Carrega modelo YOLO ──
    yolo_path = os.path.join(MODELS_DIR, "yolov8n-face.pt")
    if not os.path.exists(yolo_path):
        print(f"[ERRO] Modelo YOLO não encontrado em {yolo_path}")
        return False
    yolo = YOLO(yolo_path)

    # ── Carrega modelo de embedding ──
    recognizer = ModelFactory.create(model_name)

    # ── Conecta ao ChromaDB ──
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=model_name)
    if collection.count() == 0:
        print(f"[AVISO] Coleção '{model_name}' está vazia no ChromaDB!")

    # ── Abre o vídeo ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    paused = False

    window_name = f"Visualizador - {model_name.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t_start = time.perf_counter()

            # ── 1. Detecção YOLO ──
            predictions = yolo.predict(frame, verbose=False, conf=0.7)

            detections_to_draw = []

            for result in predictions:
                if not hasattr(result, "boxes") or result.boxes is None:
                    continue
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy

                    # Escala do crop (fator 1.5x para contexto extra)
                    scale = 1.5
                    w_bbox = x2 - x1
                    h_bbox = y2 - y1
                    cx = x1 + w_bbox / 2
                    cy = y1 + h_bbox / 2
                    w_new = w_bbox * scale
                    h_new = h_bbox * scale

                    h_img, w_img = frame.shape[:2]
                    crop_x1 = max(0, int(cx - w_new / 2))
                    crop_y1 = max(0, int(cy - h_new / 2))
                    crop_x2 = min(w_img, int(cx + w_new / 2))
                    crop_y2 = min(h_img, int(cy + h_new / 2))

                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                        continue

                    face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if face_crop.size == 0:
                        continue

                    crop_h, crop_w = face_crop.shape[:2]

                    # ── 2. Extrai embedding ──
                    embedding = recognizer.extract_embedding_with_filters(face_crop)

                    identity = "Desconhecido"
                    similarity = 0.0

                    if embedding is not None:
                        # ── 3. Busca no ChromaDB ──
                        try:
                            results = collection.query(
                                query_embeddings=[list(map(float, embedding))],
                                n_results=1,
                            )
                            if results and results.get("ids") and len(results["ids"][0]) > 0:
                                distance = results["distances"][0][0]
                                similarity = max(0.0, 1.0 - distance)
                                metadata = results["metadatas"][0][0]
                                identity = metadata.get("person", results["ids"][0][0])
                        except Exception:
                            pass

                    detections_to_draw.append({
                        "bbox": (crop_x1, crop_y1, crop_x2, crop_y2),
                        "identity": identity,
                        "similarity": similarity,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "track_id": 0,  # sem DeepSort para simplificar
                        "conf": conf,
                    })

            # ── 4. Desenha overlays ──
            for i, det in enumerate(detections_to_draw):
                draw_overlay(
                    frame,
                    det["bbox"],
                    det["identity"],
                    det["similarity"],
                    det["crop_w"],
                    det["crop_h"],
                    i + 1,
                    det["conf"],
                )

            t_end = time.perf_counter()
            processing_fps = 1.0 / max(t_end - t_start, 1e-6)

            draw_hud(frame, model_name, video_name, frame_idx, total_frames, processing_fps)

            display_frame = frame

        # ── Exibe ──
        cv2.imshow(window_name, display_frame)

        # ── Controle de teclas ──
        wait_ms = 1 if not paused else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == ord("Q"):
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord("p") or key == ord("P"):
            paused = not paused
            state = "PAUSADO" if paused else "REPRODUZINDO"
            print(f"  [{state}]")
        elif key == ord("n") or key == ord("N"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         PIPELINE DE VISUALIZAÇÃO EM TEMPO REAL             ║")
    print("║         Detecção YOLO + Embedding + ChromaDB               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Garantir modelos ONNX ──
    ensure_all_onnx_models(MODELS_DIR)

    # ── Etapa 1: Popular ChromaDB com TODOS os modelos ──
    populate_all_models(TRAINING_DIR, CHROMA_DIR)

    # ── Etapa 2: Escolher modelo de embedding ──
    model_name = choose_option("Qual modelo de embedding usar para a busca?", ALL_MODELS)
    print(f"\n  ✓ Modelo selecionado: {model_name.upper()}")

    # ── Etapa 3: Escolher vídeo(s) ──
    videos = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".mp4")]
    if not videos:
        print("[ERRO] Nenhum vídeo .mp4 encontrado em data/recordings/")
        sys.exit(1)

    video_options = ["▶ Todos os vídeos em sequência"] + videos
    choice = choose_option("Qual vídeo reproduzir?", video_options)

    if choice == video_options[0]:
        # Reproduz todos em sequência
        for video in videos:
            video_path = os.path.join(RECORDINGS_DIR, video)
            keep_going = run_visualization(model_name, video_path, video)
            if not keep_going:
                break
    else:
        video_path = os.path.join(RECORDINGS_DIR, choice)
        run_visualization(model_name, video_path, choice)

    print("\n✓ Visualização encerrada.")


if __name__ == "__main__":
    main()
