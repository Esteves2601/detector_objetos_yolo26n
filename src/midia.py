"""Mídia: imagens, vídeo e webcam. Exibição e salvamento centralizados aqui."""
import os
import time
import urllib.request
from pathlib import Path

import cv2

from .config import IMGSZ_PADRAO_HF, IMGSZ_STREAM_PADRAO, OUTPUTS_DIR, TMP_DIR
from .inference import anotar_bgr, contar, prever, resolver_imgsz_auto, resumir_contagem


def ajustar_para_tela(img, larg_max=1280, alt_max=720):
    """Redimensiona só para exibição (nunca amplia). Mantém aspecto."""
    h, w = img.shape[:2]
    escala = min(larg_max / w, alt_max / h, 1.0)
    if escala < 1.0:
        return cv2.resize(img, (int(w * escala), int(h * escala)),
                          interpolation=cv2.INTER_AREA)
    return img


def listar_imagens(pasta):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    fontes = []
    p = Path(pasta)
    for e in exts:
        fontes += [str(f) for f in p.glob(e)]
        fontes += [str(f) for f in p.glob(e.upper())]
    return sorted(fontes)


def _mostrar_janela(nome, anotada_bgr):
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome, 1280, 720)
    cv2.imshow(nome, ajustar_para_tela(anotada_bgr))
    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(nome, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def detectar_arquivo(origem, conf=0.30, iou=0.50, imgsz=None, max_det=50,
                     salvar=True, exibir=True, dir_saida=None):
    """Aceita arquivo, pasta ou lista. Salva em outputs/ (não polui a origem)."""
    from .modelo import get_model
    model = get_model()

    if isinstance(origem, (str, os.PathLike)) and Path(str(origem)).is_dir():
        fontes = listar_imagens(origem)
        if not fontes:
            print(f"Nenhuma imagem em: {origem}")
            return []
    elif isinstance(origem, (list, tuple)):
        fontes = list(origem)
    else:
        fontes = [origem]

    dir_saida = Path(dir_saida) if dir_saida else OUTPUTS_DIR
    dir_saida.mkdir(parents=True, exist_ok=True)

    saidas = []
    for fonte in fontes:
        imgsz_eff = imgsz or resolver_imgsz_auto(fonte, modo="imagem")
        t0 = time.time()
        results = prever(fonte, conf, iou, imgsz_eff, max_det, modo="imagem")
        dt_ms = (time.time() - t0) * 1000
        cont = contar(results)
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                print(f"{model.names[int(b.cls)]}: {float(b.conf):.2f}")
        print(f"[{fonte}] {dt_ms:.0f} ms -> {cont or 'nada'}")

        path_saida = None
        if salvar and isinstance(fonte, (str, os.PathLike)) and Path(str(fonte)).exists():
            p = Path(fonte)
            path_saida = str(dir_saida / f"{p.stem}_detectado{p.suffix}")
            cv2.imwrite(path_saida, anotar_bgr(results))
            print(f"Salvo em: {path_saida}")

        if exibir:
            print("Pressione 'q' na janela para continuar...")
            _mostrar_janela("YOLO26n - resultado (q fecha)", anotar_bgr(results))

        saidas.append({"arquivo": str(fonte), "contagem": cont,
                       "tempo_ms": round(dt_ms, 1), "saida": path_saida})
    return saidas


def processar_video(video_path, conf=0.30, iou=0.50, imgsz=None, max_det=30):
    """Processa vídeo e salva anotado em TMP_DIR/video. Retorna (path, resumo, totais)."""
    from .config import DEVICE
    from .modelo import get_model
    model = get_model()
    if not imgsz:
        imgsz = IMGSZ_PADRAO_HF
        print(f"Video -> imgsz={imgsz} (padrão HF)")
    out = model.predict(source=str(video_path), imgsz=int(imgsz), conf=float(conf),
                        iou=float(iou), device=DEVICE,
                        max_det=int(max_det), verbose=False, save=True,
                        project=str(TMP_DIR), name="video", exist_ok=True)
    vids = sorted((TMP_DIR / "video").glob("*.avi")) + \
        sorted((TMP_DIR / "video").glob("*.mp4"))
    total = {}
    for r in out:
        if r.boxes is not None:
            for c in r.boxes.cls.cpu().numpy().astype(int):
                total[model.names[c]] = total.get(model.names[c], 0) + 1
    return (str(vids[-1]) if vids else None, f"{len(out)} frames | {resumir_contagem(total)}", total)


def rodar_webcam_opencv(conf=0.30, iou=0.50, imgsz=None, max_det=30,
                        cam_id=0, espelhar=True, gravar=None):
    """Webcam local. 'q' sai, 's' salva frame em outputs/."""
    if not imgsz:
        imgsz = IMGSZ_STREAM_PADRAO
        print(f"Webcam -> imgsz={imgsz} automático")
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Nao foi possivel abrir a webcam {cam_id}")
        return
    writer = None
    t_prev = time.time()
    fps = 0.0
    print("[q] sair | [s] salvar frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao ler frame")
                break
            if espelhar:
                frame = cv2.flip(frame, 1)
            results = prever(frame, conf, iou, imgsz, max_det, modo="stream")
            anotada = anotar_bgr(results)
            agora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(agora - t_prev, 1e-6))
            t_prev = agora
            cv2.putText(anotada, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if gravar:
                if writer is None:
                    h, w = anotada.shape[:2]
                    writer = cv2.VideoWriter(gravar, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(anotada)
            cv2.imshow("YOLO26n - webcam (q=sair, s=salvar)", anotada)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("s"):
                nome = str(OUTPUTS_DIR / f"frame_{int(time.time())}.jpg")
                cv2.imwrite(nome, anotada)
                print(f"Frame salvo: {nome}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def baixar_exemplo():
    url = "https://ultralytics.com/images/bus.jpg"
    dst = TMP_DIR / "bus.jpg"
    if not dst.exists():
        urllib.request.urlretrieve(url, str(dst))
        print("baixado:", dst)
    return str(dst)
