"""
model_downloader.py
Utilitário responsável por garantir a presença dos modelos ONNX em data/models/.

Todos os 5 modelos ONNX (cosface, sphereface, magface, curricularface, elasticface)
utilizam o mesmo backbone (w600k_r50.onnx do InsightFace, iResNet, 512-D, 112x112).
Por isso, o módulo adota uma estratégia inteligente:

  1. Se algum dos 5 arquivos já existir localmente, copia-o para os faltantes.
  2. Só realiza download da rede se nenhum existir, e baixa apenas UMA vez.
"""

import os
import shutil
import urllib.request
import urllib.error
from typing import List

# ---------------------------------------------------------------------------
# Nomes esperados dos modelos ONNX em data/models/
# ---------------------------------------------------------------------------
ONNX_MODEL_NAMES: List[str] = [
    "cosface.onnx",
    "sphereface.onnx",
    "magface.onnx",
    "curricularface.onnx",
    "elasticface.onnx",
]

# ---------------------------------------------------------------------------
# URLs de download (todas apontam para o mesmo w600k_r50.onnx).
# Ordem: mais confiáveis primeiro.
# ---------------------------------------------------------------------------
DOWNLOAD_URLS: List[str] = [
    "https://huggingface.co/yolkailtd/face-swap-models/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx",
    "https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/w600k_r50.onnx",
    "https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx",
]


def _download_file(url: str, dest_path: str) -> bool:
    """
    Tenta fazer o download de `url` salvando em `dest_path`.
    Retorna True em caso de sucesso, False caso contrário.
    """
    try:
        print(f"    Baixando de: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as response:
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


def _find_existing_model(models_dir: str) -> str | None:
    """
    Procura um dos modelos ONNX já existentes em `models_dir`.
    Retorna o caminho absoluto do primeiro encontrado, ou None.
    """
    for name in ONNX_MODEL_NAMES:
        path = os.path.join(models_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def _download_base_model(dest_path: str) -> bool:
    """
    Tenta baixar o modelo base (w600k_r50.onnx) de uma das URLs disponíveis.
    Retorna True se bem-sucedido.
    """
    for url in DOWNLOAD_URLS:
        if _download_file(url, dest_path):
            return True
    return False


def ensure_model(model_filename: str, models_dir: str) -> str:
    """
    Garante que `model_filename` existe em `models_dir`.

    Estratégia:
      1. Se já existir, retorna imediatamente.
      2. Se outro modelo ONNX do grupo já existir, copia-o.
      3. Caso contrário, baixa da rede.

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
        Se o modelo não puder ser obtido de nenhuma forma.
    """
    dest_path = os.path.join(models_dir, model_filename)

    if os.path.exists(dest_path):
        return dest_path

    if model_filename not in ONNX_MODEL_NAMES:
        raise FileNotFoundError(
            f"Modelo '{model_filename}' não encontrado em '{models_dir}' e não "
            f"possui URLs de download configuradas."
        )

    os.makedirs(models_dir, exist_ok=True)

    # Tenta copiar de um modelo existente (mesmo backbone)
    existing = _find_existing_model(models_dir)
    if existing:
        print(f"[COPY] Copiando '{os.path.basename(existing)}' -> '{model_filename}'")
        shutil.copy2(existing, dest_path)
        print(f"[OK] Modelo '{model_filename}' pronto em: {dest_path}")
        return dest_path

    # Nenhum modelo existe ainda — precisa baixar da rede
    print(f"\n[DOWNLOAD] Modelo '{model_filename}' não encontrado. Iniciando download automático...")

    if _download_base_model(dest_path):
        print(f"[OK] Modelo '{model_filename}' salvo em: {dest_path}")
        return dest_path

    raise FileNotFoundError(
        f"Não foi possível baixar o modelo '{model_filename}' a partir das URLs configuradas.\n"
        f"Por favor, baixe o modelo manualmente e salve em: {dest_path}\n"
        f"URLs tentadas:\n"
        + "\n".join(f"  - {u}" for u in DOWNLOAD_URLS)
    )


def ensure_all_onnx_models(models_dir: str) -> None:
    """
    Garante que todos os modelos ONNX necessários estejam presentes em `models_dir`.

    Estratégia otimizada:
      1. Verifica quais estão faltando.
      2. Se algum já existir, copia para os faltantes (zero download).
      3. Se nenhum existir, baixa UMA vez e copia para os demais.

    Parâmetros
    ----------
    models_dir : str
        Caminho absoluto para o diretório data/models/.
    """
    print("\n=== Verificando modelos ONNX em data/models/ ===")
    missing = [
        name for name in ONNX_MODEL_NAMES
        if not os.path.exists(os.path.join(models_dir, name))
    ]

    if not missing:
        print("Todos os modelos ONNX já estão presentes.\n")
        return

    print(f"Modelos faltando: {', '.join(missing)}")
    os.makedirs(models_dir, exist_ok=True)

    # Verifica se algum modelo do grupo já existe para copiar
    existing = _find_existing_model(models_dir)

    if existing:
        # Copia o existente para todos os faltantes
        for model_name in missing:
            dest = os.path.join(models_dir, model_name)
            print(f"[COPY] Copiando '{os.path.basename(existing)}' -> '{model_name}'")
            shutil.copy2(existing, dest)
            print(f"[OK] '{model_name}' pronto.")
    else:
        # Nenhum existe — baixa o primeiro e copia para os demais
        first = missing[0]
        first_path = os.path.join(models_dir, first)

        print(f"\n[DOWNLOAD] Baixando modelo base como '{first}'...")
        if not _download_base_model(first_path):
            raise RuntimeError(
                f"Não foi possível baixar o modelo ONNX base.\n"
                f"Por favor, baixe manualmente de uma das URLs abaixo e salve em "
                f"'{models_dir}' com o nome de qualquer modelo esperado:\n"
                + "\n".join(f"  - {u}" for u in DOWNLOAD_URLS)
            )
        print(f"[OK] '{first}' salvo.")

        # Copia para os demais faltantes
        for model_name in missing[1:]:
            dest = os.path.join(models_dir, model_name)
            print(f"[COPY] Copiando '{first}' -> '{model_name}'")
            shutil.copy2(first_path, dest)
            print(f"[OK] '{model_name}' pronto.")

    print("=== Todos os modelos ONNX prontos. ===\n")
