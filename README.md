# Detector de Objetos YOLO26n — Multi-plataforma Offline

Detector de objetos baseado no modelo oficial **Ultralytics/YOLO26 (variante nano `yolo26n.pt`)**, com suporte a **três frentes de execução 100% offline**: desktop (Windows), mobile nativo (Android) e PWA (navegador).

---

## O que o projeto faz

Detecção de objetos em arquivos (imagem/vídeo) e via webcam usando o modelo **YOLO26n** da Ultralytics, com três opções de uso:

| Plataforma | Como roda | Recursos |
|------------|-----------|----------|
| **Desktop (Windows)** | Executável `.exe` (clique duplo) | Webcam com detecção frame a frame, pausar/retomar, salvar frame em pasta escolhida, abrir imagem/vídeo do disco, resultados salvos em `outputs/` |
| **Mobile Nativo (Android)** | APK instalável (offline total) | Câmera frontal/traseira (troca com botão), **captura única com detecção**, escolher da galeria, salvar na galeria (`Imagens/YOLO26n/`) |
| **PWA (Navegador)** | Servidor local + celular no mesmo Wi-Fi | Câmera (foto na hora) / galeria, resultado anotado + contagem, "Adicionar à tela inicial" instala como app |

---

## Modelo Utilizado

**Modelo base:** [`Ultralytics/YOLO26`](https://huggingface.co/Ultralytics/YOLO26) — variante **nano (`yolo26n.pt`)**  
**Fonte oficial:** Hugging Face Hub — `Ultralytics/YOLO26`  
**Task:** Detecção de objetos (80 classes COCO)

### O que foi alterado/adicionado em relação ao modelo original:

| Item | Original (Ultralytics) | Este Projeto |
|------|------------------------|--------------|
| **Formato** | PyTorch `.pt` (requer Python + dependências) | Mantido `.pt` + **exportado para ONNX** (9.3 MB, imgsz 320) para inferência mobile nativa |
| **Inferência** | API `ultralytics.YOLO` padrão | Wrapper com **auto-imgsz** (640 imagem/vídeo, 320 stream), **clamp de coordenadas**, **NMS por classe**, **auto-detecção de shape ONNX** |
| **Execução** | Só Python + dependências pesadas | **3 frentes**: Windows `.exe` standalone, Android APK nativo (ONNX Runtime), PWA via Flask |
| **Deploy** | `pip install ultralytics` + código | `.exe` standalone (PyInstaller), APK (Gradle), PWA (Flask) — **tudo offline** |
| **Interface** | CLI / Gradio opcional | **GUI Tkinter** (desktop), **App Android nativo** (Kotlin + CameraX), **PWA mobile-first** |

### Adições deste projeto (não existiam no modelo original):

1. **Auto-imgsz inteligente** — detecta resolução da entrada e escolhe 640 (imagem/vídeo) ou 320 (stream) automaticamente
2. **Parser ONNX robusto** — detecta shape de saída automaticamente (`[1,84,N]`, `[1,N,84]`, flat), faz transpose se necessário, bounds checking em coordenadas
3. **Core modularizado** (`src/`) — config, modelo lazy singleton, inferência, mídia, Gradio separados; sem side-effects no import
4. **Desktop standalone** — PyInstaller com `--collect-all torchvision` (inclui operador `nms` nativo), ícone custom, thread-safe webcam com pausar/retomar
5. **Android nativo** — CameraX + ONNX Runtime 1.17.1, **captura única com detecção**, troca de câmera frontal/traseira, salvar na galeria, `AutoCloseable` para limpeza de sessão
6. **PWA mobile-first** — Flask + HTML com `capture="environment"`, manifest + service worker, instala como app nativo
7. **Organização** — pastas `data/exemplos/`, `outputs/`, `assets/`, `tools/`; `.gitignore` limpo; atalhos `.vbs`/`.bat` para Windows

---

## Estrutura do Projeto

```
.
├── app_gui.py              # GUI Desktop (Tkinter) — webcam tempo real | escolher arquivo
├── app_web.py              # Servidor PWA (Flask) — acesso via Wi-Fi no celular
├── detector_objeto.py      # CLI + fachada de compatibilidade (--modo imagem|webcam|video|benchmark)
├── dist/                   # Executável Windows (DetectorYOLO26n.exe) — gerado pelo PyInstaller
├── android/                # Projeto Android nativo (Kotlin + Gradle)
│   ├── app/src/main/
│   │   ├── assets/
│   │   │   ├── yolo26n.onnx      # Modelo ONNX (9.3 MB, imgsz 320)
│   │   │   └── labels.txt        # 80 classes COCO
│   │   ├── java/org/yolo26/detector/
│   │   │   ├── MainActivity.kt   # CameraX + captura única + troca câmera
│   │   │   ├── YoloDetector.kt   # ONNX Runtime + parser robusto
│   │   │   └── OverlayView.kt    # Desenha caixas sobre preview
│   │   └── res/                  # Layout, ícones, temas
│   └── build.gradle              # ONNX Runtime 1.17.1, CameraX 1.3.1
├── assets/                 # Ícones (icon.png/.ico, icon-192/512.png p/ PWA/Android)
├── data/exemplos/          # Imagens de teste
├── outputs/                # Resultados salvos (ignorado no git)
├── exports/                # Modelos exportados (ONNX)
├── mobile_app/             # Protótipo Kivy (cliente-servidor) — legado
├── src/                    # Core Python modularizado
│   ├── __init__.py
│   ├── config.py           # Constantes centralizadas (HF, imgsz, device, pastas)
│   ├── modelo.py           # Singleton lazy do YOLO (fuse + warmup)
│   ├── inference.py        # prever(), contar(), anotar(), auto-imgsz
│   ├── midia.py            # Imagem/vídeo/webcam + exibição/salvamento
│   └── gradio_app.py       # Interface Gradio opcional
├── tools/
│   ├── export_mobile.py    # Exporta .pt → ONNX/TFLite/NCNN
│   ├── make_icon.py        # Gera ícones do app
│   └── validar_onnx.py     # Valida inferência ONNX local
├── requirements.txt        # Dependências (ultralytics, torch, opencv, onnxruntime, flask opcional)
└── .gitignore
```

---

## Como Usar

### 1. Desktop (Windows) — Mais simples
```bash
# Opção A: Executável pronto (clique duplo)
dist/DetectorYOLO26n.exe

# Opção B: Via Python
pip install -r requirements.txt
python app_gui.py
```

**Funcionalidades Desktop:** webcam com detecção frame a frame (pausar/retomar), salvar frame em pasta escolhida, abrir imagem/vídeo do disco, resultados salvos em `outputs/`

### 2. Mobile Nativo (Android) — Offline total
1. Abra a pasta `android/` no **Android Studio**
2. Aguarde o Gradle Sync → **Run ▶** (celular via USB) ou **Build > Build APK(s)**
3. Instale o APK no celular → permita câmera → use

**Funcionalidades Android:** câmera frontal/traseira (troca com botão), **captura única com detecção**, escolher da galeria, salvar na galeria (`Imagens/YOLO26n/`)

> O modelo `yolo26n.onnx` já está embutido em `app/src/main/assets/`

### 3. PWA (Navegador do celular) — Via Wi-Fi
```bash
pip install flask
python app_web.py
```
No celular (mesmo Wi-Fi): abra `http://<IP-DO-PC>:5000` → menu do navegador > **Adicionar à tela inicial**

**Funcionalidades PWA:** câmera (foto na hora) / galeria, resultado anotado + contagem, "Adicionar à tela inicial" instala como app

### 4. CLI (Terminal)
```bash
python detector_objeto.py --modo imagem --origem ./data/exemplos/_MG_3910.JPG
python detector_objeto.py --modo webcam
python detector_objeto.py --modo video --origem ./video.mp4
python detector_objeto.py --modo benchmark
```

---

## 🔧 Recompilar / Gerar Artefatos

| Artefato | Comando |
|----------|---------|
| **Executável Windows** | `py -m PyInstaller --noconfirm DetectorYOLO26n.spec` |
| **APK Android** | Abrir `android/` no Android Studio → Build > Build APK(s) |
| **Exportar ONNX** | `python tools/export_mobile.py --formato onnx --imgsz 320` |
| **Validar ONNX** | `python tools/validar_onnx.py` |
| **Gerar ícones** | `python tools/make_icon.py` |

---

## Dependências Principais

```txt
# Core (obrigatório)
ultralytics>=8.3
torch>=2.0
opencv-python>=4.8
numpy>=1.24
Pillow>=10
onnx>=1.15
onnxruntime>=1.17

# Opcional
flask>=3.0          # Para app_web.py (PWA)
# gradio>=4         # Para detector_objeto.py --modo gradio
```

---

## ⚠️ Nota sobre Tempo Real no Android

A funcionalidade de **tempo real contínuo (preview com detecção ao vivo)** foi **removida da versão Android** devido a instabilidades no parser ONNX e problemas de performance no `ImageAnalysis` do CameraX. 

A versão Android atual foca em **captura única confiável**: você aponta a câmera, aperta "Capturar", e o app faz a detecção naquele frame, mostrando o resultado com caixas e contagem. Mantém-se a troca de câmera (frontal/traseira), galeria e salvamento na galeria.

O **Desktop (Windows)** mantém a detecção frame a frame com pausar/retomar, e o **PWA** funciona com captura única via navegador.

---

## Licença

O modelo **YOLO26** é da **Ultralytics** (licença AGPL-3.0).  
Este código wrapper é livre para uso educacional e pesquisa.

---

## Autor

**Esteves2601** — [GitHub](https://github.com/Esteves2601)

> Projeto criado para demonstrar deploy multi-plataforma offline de modelo YOLO (desktop + mobile nativo + PWA) a partir de um único modelo base.
