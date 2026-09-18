"""Inferência: única chamada ao modelo + helpers de contagem/anotação."""
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import IMGSZ_PADRAO_HF, IMGSZ_STREAM_PADRAO, STRIDE


def _arredondar_stride(v, stride=STRIDE):
    return max(stride, int(round(v / stride) * stride))


def resolver_imgsz_auto(source, modo="imagem"):
    """imgsz automático: imagem/video -> 640 (padrão HF), stream -> 320 (FPS)."""
    from .config import IMGSZ_PADRAO_HF as HF
    from .config import IMGSZ_STREAM_PADRAO as ST
    padrao = HF if modo in ("imagem", "video") else ST
    try:
        if isinstance(source, np.ndarray):
            h, w = source.shape[:2]
        elif isinstance(source, Image.Image):
            w, h = source.size
        elif isinstance(source, (str, os.PathLike)) and Path(str(source)).exists():
            img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if img is None:
                return padrao
            h, w = img.shape[:2]
        else:
            return padrao  # URL, webcam, etc.: usa padrão do modo
        maior = max(h, w)
        return padrao if maior > padrao else _arredondar_stride(maior)
    except Exception as e:
        print(f"Auto-imgsz falhou ({e}), usando {padrao}")
        return padrao


def prever(source, conf=0.30, iou=0.50, imgsz=None, max_det=30, modo="imagem"):
    """Chamada única ao modelo. imgsz=None/0 = AUTO."""
    from .config import DEVICE
    from .modelo import get_model
    if not imgsz:
        imgsz = resolver_imgsz_auto(source, modo=modo)
    return get_model().predict(
        source=source, imgsz=int(imgsz), conf=float(conf), iou=float(iou),
        device=DEVICE, max_det=int(max_det),
        agnostic_nms=False, verbose=False,
    )


def contar(results, names=None):
    """Conta objetos por classe. Recebe names para não depender de global."""
    if names is None:
        from .modelo import get_model
        names = get_model().names
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {}
    ids = r.boxes.cls.cpu().numpy().astype(int)
    return dict(Counter(names[i] for i in ids))


def resumir_contagem(cont):
    return ", ".join(f"{k}: {v}" for k, v in sorted(cont.items())) or "nada detectado"


def anotar_rgb(results, line_width=None):
    try:
        bgr = results[0].plot(line_width=line_width) if line_width else results[0].plot()
    except TypeError:
        bgr = results[0].plot()
    return bgr[:, :, ::-1]


def anotar_bgr(results, line_width=None):
    try:
        return results[0].plot(line_width=line_width) if line_width else results[0].plot()
    except TypeError:
        return results[0].plot()


def detectar(frame, conf=0.30, iou=0.50, imgsz=None, max_det=30):
    """Aceita numpy ou PIL. Retorna (img_rgb, info, contagem)."""
    if frame is None:
        return None, "Nenhuma imagem recebida.", {}
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    if not imgsz:
        imgsz = resolver_imgsz_auto(frame, modo="stream")
    t0 = time.time()
    results = prever(frame, conf, iou, imgsz, max_det, modo="stream")
    dt_ms = (time.time() - t0) * 1000
    fps = 1000.0 / max(dt_ms, 1e-6)
    cont = contar(results)
    total = sum(cont.values())
    img = anotar_rgb(results, line_width=max(1, int(imgsz) // 200))
    return img, f"{dt_ms:.0f} ms ({fps:.1f} FPS) | {total} obj | {resumir_contagem(cont)}", cont
