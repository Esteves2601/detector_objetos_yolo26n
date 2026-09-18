"""Pacote core do detector YOLO26n."""
from .config import (
    HF_REPO, HF_FILE, HF_ID,
    IMGSZ_PADRAO_HF, IMGSZ_STREAM_PADRAO, STRIDE,
    DEVICE, IS_CUDA, USE_HALF,
    BASE_DIR, TMP_DIR, OUTPUTS_DIR, DATA_EXEMPLOS_DIR,
    EXT_IMAGEM, EXT_VIDEO,
)
from .modelo import get_model, carregar_modelo_hf
from .inference import prever, contar, anotar_rgb, detectar, resolver_imgsz_auto, resumir_contagem
from .midia import ajustar_para_tela, detectar_arquivo, processar_video, rodar_webcam_opencv, baixar_exemplo

__all__ = [
    "HF_REPO", "HF_FILE", "HF_ID",
    "IMGSZ_PADRAO_HF", "IMGSZ_STREAM_PADRAO", "STRIDE",
    "DEVICE", "IS_CUDA", "USE_HALF",
    "BASE_DIR", "TMP_DIR", "OUTPUTS_DIR", "DATA_EXEMPLOS_DIR",
    "EXT_IMAGEM", "EXT_VIDEO",
    "get_model", "carregar_modelo_hf",
    "prever", "contar", "anotar_rgb", "detectar",
    "resolver_imgsz_auto", "resumir_contagem",
    "ajustar_para_tela", "detectar_arquivo", "processar_video",
    "rodar_webcam_opencv", "baixar_exemplo",
]
