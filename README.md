# Detector de Objetos YOLO26n

Detecção de objetos com modelo oficial `Ultralytics/YOLO26` (variante nano `yolo26n.pt`).
Duas entradas: **Tkinter GUI** (`app_gui.py`) e **CLI** (`detector_objeto.py`), com
imagem, vídeo, webcam ao vivo e Gradio opcional.

## O que o projeto faz

- `prever()` — chamada única ao YOLO com `imgsz` automático (imagem/vídeo → 640, stream → 320).
- `contar()` / `resumir_contagem()` — contagem por classe.
- `detectar_arquivo()` — imagem ou pasta → anota e salva em `outputs/`.
- `processar_video()` — vídeo → anotado em temp + resumo de frames/classes.
- `rodar_webcam_opencv()` / GUI — webcam espelhada com FPS, salvar frame com `s`.
- `rodar_gradio()` — interface web opcional (imagem, streaming, vídeo).

## Estrutura

```
app_gui.py           GUI Tkinter (câmera ao vivo | escolher arquivo)
app_web.py           Servidor web mobile/PWA — celular acessa via Wi-Fi
detector_objeto.py   CLI + fachada de compatibilidade (--modo imagem|webcam|video|gradio|benchmark)
mobile_app/          Cliente Android (Kivy + buildozer.spec) — envia foto ao app_web.py
assets/              Ícones do app (icon.png/.ico, icon-192/512 p/ PWA e Android)
tools/               make_icon.py, export_mobile.py (TFLite/NCNN p/ inferência on-device)
src/
  config.py          constantes (HF, imgsz, device, pastas, extensões)
  modelo.py          singleton lazy do YOLO (sem side-effect no import)
  inference.py       prever/contar/anotar/detectar + auto-imgsz
  midia.py           imagem/vídeo/webcam + exibição/salvamento
  gradio_app.py      interface web opcional
data/exemplos/       imagens de exemplo
outputs/             resultados `*_detectado.*` e frames (ignorado no git)
requirements.txt
```

## Uso

```bash
pip install -r requirements.txt
python app_gui.py
python detector_objeto.py --modo imagem --origem ./data/exemplos/_MG_3910.JPG --sem-exibir
python detector_objeto.py --modo webcam
python detector_objeto.py --modo video --origem ./video.mp4
python detector_objeto.py --modo benchmark
pip install "gradio>=4"; python detector_objeto.py --modo gradio
```

## Celular (2 opções)

**A — PWA, sem instalar nada (recomendado):**
```bash
pip install flask
py app_web.py
```
No celular (mesmo Wi-Fi), abra o `http://<IP>:5000` impresso no terminal,
depois menu do navegador > **Adicionar à tela inicial** (ícone próprio, tela cheia).
Botões de câmera (foto na hora) e galeria, com resultado anotado + contagem.

**B — APK nativo offline (de verdade, sem servidor):**
`android/` é um projeto Android completo (Kotlin + CameraX + ONNX Runtime).
O modelo `yolo26n.onnx` (9,3 MB, imgsz 320) já está em `app/src/main/assets/`
e foi validado aqui via onnxruntime (`tools/validar_onnx.py`).
Para gerar o APK: abra a pasta `android/` no **Android Studio** → aguarde o
Sync → **Run ▶** num celular via USB (ou Build > Build APK).
Ícone do app já aplicado (`@drawable/icon`, o olho-detector).
`mobile_app/` (Kivy) é o protótipo antigo cliente-servidor — desconsidere.

## Executável Windows

`dist/DetectorYOLO26n.exe` — clique duplo, sem console, com ícone próprio
(`assets/icon.ico`, olho-detector + bounding box). Otimizações aplicadas:
exclusão de gradio/matplotlib/jupyter/testes/tensorboard, modelo e `data/`
embutidos, pastas resolvidas para o lado do `.exe` (modo frozen).
Recompilar: `py -m PyInstaller --noconfirm DetectorYOLO26n.spec`

## Redundâncias removidas nesta revisão

- `Arquivos/detector_objeto.py` era cópia antiga (sem auto-imgsz) → removida; imagens foram para `data/exemplos/`, resultados para `outputs/`.
- `Arquivos/requirements.txt` duplicado → removido.
- `matplotlib` não era usado → saiu do `requirements.txt`; `gradio` virou opcional.
- Modelo não carrega mais no `import` → `src/modelo.py:get_model()` com fuse + warmup uma vez.
- Contagem de classes triplicada → `src/inference.py:contar()` única.
- `app_gui.py`: variável `rgb` morta removida; `salvar_frame` recapturava outro frame (bug) → agora salva o frame exibido; `messagebox`/status em thread via `after()`; janela de vídeo e save centralizados em `outputs/`.
- Saídas `*_detectado` não poluem mais a pasta de origem → sempre em `outputs/`.
- `DEVICE` misturava `int`/`str` → `IS_CUDA` + `DEVICE` documentados em `src/config.py`.
