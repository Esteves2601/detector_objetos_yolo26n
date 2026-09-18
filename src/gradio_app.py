"""Interface Gradio (opcional). Import pesado fica isolado aqui."""


def rodar_gradio(share=False):
    import gradio as gr

    from .inference import detectar
    from .midia import processar_video

    with gr.Blocks(title="YOLO26n — Arquivo ou Webcam") as demo:
        gr.Markdown("## Detector YOLO26n — imagem salva, webcam ou vídeo")
        with gr.Row():
            conf = gr.Slider(0.05, 0.95, value=0.30, step=0.05, label="Confianca")
            iou = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="IoU (NMS)")
            imgsz = gr.Slider(160, 640, value=320, step=32, label="imgsz")
            max_det = gr.Slider(1, 100, value=30, step=1, label="Max deteccoes")
        with gr.Tab("Imagem (arquivo ou webcam)"):
            inp_img = gr.Image(sources=["upload", "webcam"], type="numpy",
                               label="Envie arquivo ou tire foto")
            btn_img = gr.Button("Detectar", variant="primary")
            out_img = gr.Image(label="Resultado")
            out_txt = gr.Textbox(label="Tempo / resumo")
            out_cnt = gr.Label(label="Contagem por classe")
            btn_img.click(detectar, [inp_img, conf, iou, imgsz, max_det],
                          [out_img, out_txt, out_cnt])
        with gr.Tab("Webcam ao vivo (streaming)"):
            inp_live = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                                label="Webcam ao vivo")
            out_live = gr.Image(label="Deteccao ao vivo")
            out_live_txt = gr.Textbox(label="Info")
            out_live_cnt = gr.Label(label="Contagem")
            inp_live.stream(lambda f, c, i, s, m: detectar(f, c, i, s, m),
                            [inp_live, conf, iou, imgsz, max_det],
                            [out_live, out_live_txt, out_live_cnt],
                            time_limit=60, stream_every=0.1, concurrency_limit=2)
        with gr.Tab("Video (arquivo)"):
            inp_vid = gr.Video(sources=["upload", "webcam"], label="Envie um video")
            btn_vid = gr.Button("Processar video", variant="primary")
            out_vid = gr.Video(label="Video anotado")
            out_vid_txt = gr.Textbox(label="Resumo")
            out_vid_cnt = gr.Label(label="Totais")
            btn_vid.click(processar_video, [inp_vid, conf, iou, imgsz, max_det],
                          [out_vid, out_vid_txt, out_vid_cnt])
    demo.launch(share=share)
