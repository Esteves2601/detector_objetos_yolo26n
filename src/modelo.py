"""Carregamento lazy do modelo — sem side-effect no import."""
import numpy as np

from .config import DEVICE, HF_FILE, HF_ID, HF_REPO

_model = None
_warmup_ok = False


def carregar_modelo_hf(hf_id=HF_ID, fallback=HF_FILE):
    """Baixa via huggingface_hub com fallback para peso local/embutido."""
    from ultralytics import YOLO

    try:
        from huggingface_hub import hf_hub_download
        peso_local = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        return YOLO(peso_local)
    except Exception as e1:
        print(f"HF falhou ({e1}), usando fallback {fallback} ...")
    # .exe: tenta o peso embutido (_MEIPASS) ou ao lado do executável
    try:
        from .config import recurso
        return YOLO(recurso(fallback))
    except Exception:
        return YOLO(fallback)


def get_model():
    """Singleton: carrega + fuse + warmup uma única vez, sob demanda."""
    global _model, _warmup_ok
    if _model is not None:
        return _model

    _model = carregar_modelo_hf()
    print(f"Modelo: {HF_ID} | classes={len(_model.names)} | task={_model.task}")
    try:
        _model.fuse()
    except Exception as e:
        print("fuse() pulado:", e)

    if not _warmup_ok:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(2):
            _model.predict(source=dummy, imgsz=160, conf=0.3,
                           device=DEVICE, verbose=False)
        _warmup_ok = True
        print("Modelo pronto + warmup OK.")
    return _model
