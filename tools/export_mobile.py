"""Exporta yolo26n.pt para formato leve de celular (TFLite/N CNN).

Uso:
    py tools/export_mobile.py --formato tflite   # Android (recomendado)
    py tools/export_mobile.py --formato ncnn     # Android alternativo
Requer: pip install onnx tensorflow-cpu  (só na hora de exportar)
Saída: pasta exports/ — embutir no app nativo com inferência on-device.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--formato", default="tflite", choices=["tflite", "ncnn", "onnx"])
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args(argv)
    out = Path("exports")
    out.mkdir(exist_ok=True)
    model = YOLO("yolo26n.pt")
    model.export(format=args.formato, imgsz=args.imgsz, project=str(out))
    print(f"Exportado p/ {out}/  (formato={args.formato})")


if __name__ == "__main__":
    main()
