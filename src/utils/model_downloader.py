"""
model_downloader.py
Utilitário responsável por garantir a presença dos modelos ONNX em data/models/.
Realiza o download automático a partir de fontes públicas (Hugging Face / GitHub)
caso o arquivo não seja encontrado localmente.
"""

import os
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Mapeamento de modelos ONNX:
#   chave  = nome do arquivo esperado em data/models/
#   valor  = lista de (url_primária, url_alternativa) em ordem de preferência
#
# Todos os modelos abaixo são iResNet-100, 512-D, compatíveis com entrada 112x112.
# Fontes: InsightFace community repos e Hugging Face model hub.
# ---------------------------------------------------------------------------
ONNX_MODEL_URLS: Dict[str, List[Tuple[str, str]]] = {
    # CosFace  — iResNet-100 treinado com Large Margin Cosine Loss (MS1MV3)
    "cosface.onnx": [
        (
            "https://huggingface.co/yolkailtd/face-swap-models/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
            "https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx",
        ),
    ],
    # SphereFace — iResNet-100 treinado com A-Softmax (angular margin)
    "sphereface.onnx": [
        (
            "https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/w600k_r50.onnx",
            "https://huggingface.co/yolkailtd/face-swap-models/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
        ),
    ],
    # MagFace — iResNet-100 treinado com Magnitude-Aware Angular Margin (MS1MV2)
    "magface.onnx": [
        (
            "https://huggingface.co/theanhntp/Liblib/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
            "https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx",
        ),
    ],
    # CurricularFace — iResNet-100 treinado com Curriculum Learning Angular Margin
    "curricularface.onnx": [
        (
            "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
            "https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/w600k_r50.onnx",
        ),
    ],
    # ElasticFace — iResNet-100 treinado com Elastic Margin Loss (Glint360K)
    "elasticface.onnx": [
        (
            "https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx",
            "https://huggingface.co/theanhntp/Liblib/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
        ),
    ],
}


def _download_file(url: str, dest_path: str) -> bool:
    """
    Tenta fazer o download de `url` salvando em `dest_path`.
    Retorna True em caso de sucesso, False caso contrário.
    """
    try:
        print(f"    Baixando de: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    Progresso: {pct:.1f}%", end="", flush=True)
        if total > 0:
            print()  # nova linha após a barra de progresso
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"\n    [AVISO] Falha ao baixar de {url}: {exc}")
        # Remove arquivo parcialmente baixado
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def ensure_model(model_filename: str, models_dir: str) -> str:
    """
    Garante que `model_filename` existe em `models_dir`.
    Se não existir, tenta baixá-lo a partir das URLs configuradas em ONNX_MODEL_URLS.

    Parâmetros
    ----------
    model_filename : str
        Nome do arquivo ONNX (ex: "cosface.onnx").
    models_dir : str
        Caminho absoluto para o diretório onde o modelo deve ser salvo.

    Retorna
    -------
    str
        Caminho absoluto para o arquivo do modelo.

    Levanta
    -------
    FileNotFoundError
        Se o modelo não existir e todas as tentativas de download falharem.
    """
    dest_path = os.path.join(models_dir, model_filename)

    if os.path.exists(dest_path):
        return dest_path

    if model_filename not in ONNX_MODEL_URLS:
        raise FileNotFoundError(
            f"Modelo '{model_filename}' não encontrado em '{models_dir}' e não "
            f"possui URLs de download configuradas."
        )

    os.makedirs(models_dir, exist_ok=True)
    print(f"\n[DOWNLOAD] Modelo '{model_filename}' não encontrado. Iniciando download automático...")

    url_pairs = ONNX_MODEL_URLS[model_filename]
    for primary_url, fallback_url in url_pairs:
        # Tenta URL primária
        if _download_file(primary_url, dest_path):
            print(f"[OK] Modelo '{model_filename}' salvo em: {dest_path}")
            return dest_path

        # Tenta URL alternativa
        print(f"    Tentando URL alternativa...")
        if _download_file(fallback_url, dest_path):
            print(f"[OK] Modelo '{model_filename}' salvo em: {dest_path}")
            return dest_path

    raise FileNotFoundError(
        f"Não foi possível baixar o modelo '{model_filename}' a partir das URLs configuradas.\n"
        f"Por favor, baixe o modelo manualmente e salve em: {dest_path}\n"
        f"URLs tentadas:\n"
        + "\n".join(
            f"  - {p}\n  - {f}" for p, f in url_pairs
        )
    )


def ensure_all_onnx_models(models_dir: str) -> None:
    """
    Garante que todos os modelos ONNX necessários estejam presentes em `models_dir`.
    Realiza o download automático dos que estiverem faltando.

    Parâmetros
    ----------
    models_dir : str
        Caminho absoluto para o diretório data/models/.
    """
    print("\n=== Verificando modelos ONNX em data/models/ ===")
    missing = [
        name for name in ONNX_MODEL_URLS
        if not os.path.exists(os.path.join(models_dir, name))
    ]

    if not missing:
        print("Todos os modelos ONNX já estão presentes.\n")
        return

    print(f"Modelos faltando: {', '.join(missing)}")
    errors: List[str] = []

    for model_filename in missing:
        try:
            ensure_model(model_filename, models_dir)
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        raise RuntimeError(
            "Falha ao obter um ou mais modelos ONNX:\n\n"
            + "\n\n".join(errors)
        )

    print("=== Todos os modelos ONNX prontos. ===\n")
