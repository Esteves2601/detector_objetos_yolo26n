"""Valida o modelo ONNX mobile: inferência via onnxruntime + NMS manual."""
import sys
from pathlib import Path

import cv2
import numpy as np

ONNX = Path("exports/yolo26n.onnx")
IMG = Path("data/exemplos/_MG_3910.JPG")
IMGSZ = 320
CONF = 0.30
IOU = 0.50


def letterbox(img, n=IMGSZ):
    h, w = img.shape[:2]
    s = min(n / w, n / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((n, n, 3), 114, np.uint8)
    out[(n - nh) // 2:(n - nh) // 2 + nh, (n - nw) // 2:(n - nw) // 2 + nw] = r
    return out, s, (n - nw) // 2, (n - nh) // 2


def nms(boxes, scores, iou=IOU):
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx):
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
        ovr = inter / np.maximum(a1 + a2 - inter, 1e-6)
        idx = idx[1:][ovr <= iou]
    return keep


def main():
    import onnxruntime as ort
    names = eval(Path("mobile_labels.txt").read_text()) if Path("mobile_labels.txt").exists() else None
    img0 = cv2.imread(str(IMG))
    assert img0 is not None, "imagem exemplo não encontrada"
    img, s, px, py = letterbox(img0)
    x = (img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    print("output:", out.shape, out.dtype)
    o = out[0]
    if o.shape[0] < o.shape[1]:  # (84,N) -> (N,84)
        o = o.T
    nc = o.shape[1] - 4
    boxes_cxcy, scores_all = o[:, :4], o[:, 4:]
    cls = scores_all.argmax(1)
    conf = scores_all.max(1)
    m = conf >= CONF
    boxes_cxcy, cls, conf = boxes_cxcy[m], cls[m], conf[m]
    # cxcywh (modelo) -> xyxy (letterbox) -> xyxy (original)
    b = np.empty_like(boxes_cxcy)
    b[:, 0] = boxes_cxcy[:, 0] - boxes_cxcy[:, 2] / 2
    b[:, 1] = boxes_cxcy[:, 1] - boxes_cxcy[:, 3] / 2
    b[:, 2] = boxes_cxcy[:, 0] + boxes_cxcy[:, 2] / 2
    b[:, 3] = boxes_cxcy[:, 1] + boxes_cxcy[:, 3] / 2
    b = (b - np.array([px, py, px, py])) / s
    keep = nms(b, conf)
    from collections import Counter
    det = [(names[c] if names else f"cls{c}", round(float(conf[i]), 2)) for i in keep for c in [int(cls[i])]]
    print(f"{len(keep)} detecções: {dict(Counter(c for c, _ in det))}")
    for c, cf in det[:10]:
        print(f"  {c}: {cf:.2f}")
    return 0 if len(keep) else 2


if __name__ == "__main__":
    sys.exit(main())
