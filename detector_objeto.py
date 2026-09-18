"""Detector de Objetos YOLO26n — CLI + fachada de compatibilidade.

Modelo oficial: https://huggingface.co/Ultralytics/YOLO26

Uso:
    python detector_objeto.py --modo imagem --origem ./data/exemplos/foto.jpg
    python detector_objeto.py --modo webcam --cam-id 0
    python detector_objeto.py --modo gradio
    python detector_objeto.py --modo video --origem ./video.mp4
    python detector_objeto.py --modo benchmark

Núcleo real em src/ (config, modelo, inference, midia, gradio_app).
Este arquivo só re-exporta a API antiga e implementa o CLI.
"""
import argparse
import sys
import time

import numpy as np

from src.config import (
    DEVICE, HF_FILE, HF_ID, HF_REPO,
    IMGSZ_PADRAO_HF, IMGSZ_STREAM_PADRAO, STRIDE,
    TMP_DIR, OUTPUTS_DIR, IS_CUDA, USE_HALF,
    EXT_IMAGEM, EXT_VIDEO,
)
from src.modelo import carregar_modelo_hf, get_model
from src.inference import (
    anotar_rgb, contar, detectar, prever,
    resolver_imgsz_auto, resumir_contagem, anotar_bgr,
)
from src.midia import (
    ajustar_para_tela, baixar_exemplo, detectar_arquivo,
    processar_video, rodar_webcam_opencv,
)

__all__ = [
    "DEVICE", "USE_HALF", "HF_ID", "HF_REPO", "HF_FILE", "TMP_DIR",
    "OUTPUTS_DIR", "EXT_IMAGEM", "EXT_VIDEO",
    "IMGSZ_PADRAO_HF", "IMGSZ_STREAM_PADRAO", "STRIDE",
    "carregar_modelo_hf", "prever", "contar", "anotar_rgb",
    "detectar", "ajustar_para_tela", "detectar_arquivo",
    "processar_video", "rodar_webcam_opencv",
]


def benchmark():
    model = get_model()
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    sizes = [160, 320, 480, 640] if IS_CUDA else [160, 320]
    for sz in sizes:
        t0 = time.time()
        model.predict(source=dummy, imgsz=sz, conf=0.3,
                      device=DEVICE, verbose=False)
        print(f"imgsz={sz}: {(time.time() - t0) * 1000:.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO26n local (HF: Ultralytics/YOLO26)")
    ap.add_argument("--modo", choices=["imagem", "webcam", "gradio", "video", "benchmark"],
                    default="gradio")
    ap.add_argument("--origem", default=None, help="arquivo/pasta/URL (imagem ou video)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=0, help="0=AUTO (imagem->640, webcam->320)")
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--gravar", default=None, help="path mp4 p/ gravar webcam")
    ap.add_argument("--sem-exibir", action="store_true", help="nao abre janela OpenCV")
    ap.add_argument("--share", action="store_true", help="gradio com link publico")
    args = ap.parse_args(argv)

    if args.modo == "benchmark":
        benchmark()
    elif args.modo == "webcam":
        rodar_webcam_opencv(args.conf, args.iou, args.imgsz, args.max_det,
                            args.cam_id, gravar=args.gravar)
    elif args.modo == "imagem":
        origem = args.origem or baixar_exemplo()
        detectar_arquivo(origem, args.conf, args.iou, args.imgsz,
                         args.max_det, exibir=not args.sem_exibir)
    elif args.modo == "video":
        if not args.origem:
            print("Informe --origem ./video.mp4", file=sys.stderr)
            sys.exit(2)
        path, resumo, _ = processar_video(args.origem, args.conf, args.iou,
                                          args.imgsz, args.max_det)
        print(f"Video anotado: {path}\n{resumo}")
    else:
        from src.gradio_app import rodar_gradio
        rodar_gradio(share=args.share)


if __name__ == "__main__":
    main()
