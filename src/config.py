"""Constantes centralizadas — único lugar de configuração."""
from pathlib import Path
import tempfile

import torch

# --- Modelo ---
HF_REPO = "Ultralytics/YOLO26"
HF_FILE = "yolo26n.pt"  # yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
HF_ID = f"{HF_REPO}/{HF_FILE}"

# --- Inferência ---
IMGSZ_PADRAO_HF = 640      # imagem/vídeo: padrão de treino/validação do YOLO26
IMGSZ_STREAM_PADRAO = 320  # webcam/streaming: menor = mais FPS em CPU
STRIDE = 32                # YOLO exige múltiplo de 32

# --- Device (tipo único: int 0 = cuda:0, "cpu" = CPU) ---
IS_CUDA = torch.cuda.is_available()
DEVICE = 0 if IS_CUDA else "cpu"
USE_HALF = IS_CUDA
if IS_CUDA:
    torch.backends.cudnn.benchmark = True

# --- Pastas ---
def _base_dir():
    """No .exe (PyInstaller frozen/_MEIPASS), usa a pasta do executável."""
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def recurso(nome):
    """Acha arquivo embutido no .exe (_MEIPASS) ou no projeto."""
    import sys
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and (Path(meipass) / nome).exists():
        return str(Path(meipass) / nome)
    local = BASE_DIR / nome
    if local.exists():
        return str(local)
    return nome


BASE_DIR = _base_dir()
DATA_EXEMPLOS_DIR = BASE_DIR / "data" / "exemplos"
OUTPUTS_DIR = BASE_DIR / "outputs"
TMP_DIR = Path(tempfile.gettempdir()) / "yolo26_out"

for _d in (DATA_EXEMPLOS_DIR, OUTPUTS_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Extensões ---
EXT_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv")
