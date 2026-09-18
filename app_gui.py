"""Sistema YOLO26n — Desktop: Câmera ao vivo (tempo real) | Escolher arquivo.
Roda sem terminal: apenas execute este arquivo no PyCharm (Run).
"""
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from detector_objeto import (
    contar, prever, ajustar_para_tela,
    processar_video, OUTPUTS_DIR,
    IMGSZ_PADRAO_HF, IMGSZ_STREAM_PADRAO,
    EXT_IMAGEM, EXT_VIDEO,
)

# ---------- Tema ----------
BG = "#0f172a"
CARD = "#1e293b"
CARD_BORDA = "#334155"
TEXTO = "#f1f5f9"
SUAVE = "#94a3b8"
VERDE = "#22c55e"
VERDE_HOVER = "#16a34a"
AZUL = "#3b82f6"
AZUL_HOVER = "#2563eb"
LARANJA = "#f59e0b"
LARANJA_HOVER = "#d97706"
ROXO = "#8b5cf6"
ROXO_HOVER = "#7c3aed"
FONTE_TITULO = ("Segoe UI", 18, "bold")
FONTE_SUB = ("Segoe UI", 10)
FONTE_CARD_TITULO = ("Segoe UI", 12, "bold")
FONTE_CARD_DESC = ("Segoe UI", 9)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Detector YOLO26n — HF Ultralytics/YOLO26")
        root.geometry("660x460")
        root.resizable(False, False)
        root.configure(bg=BG)
        try:
            root.attributes("-alpha", 1.0)
        except Exception:
            pass

        root.update_idletasks()
        x = (root.winfo_screenwidth() - 660) // 2
        y = (root.winfo_screenheight() - 460) // 2
        root.geometry(f"660x460+{x}+{y}")

        # ---- topo ----
        topo = tk.Frame(root, bg=BG)
        topo.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(topo, text="◉  Detector de Objetos", bg=BG, fg=VERDE,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(topo, text="YOLO26n", bg=BG, fg=TEXTO,
                 font=FONTE_TITULO).pack(anchor="w")
        tk.Label(topo, text="Ultralytics / YOLO26  •  modelo nano  •  imagem, vídeo e webcam",
                 bg=BG, fg=SUAVE, font=FONTE_SUB).pack(anchor="w", pady=(2, 0))

        badge = tk.Frame(topo, bg="#052e16", highlightbackground="#166534",
                         highlightthickness=1)
        badge.pack(anchor="w", pady=(8, 0))
        tk.Label(badge, text="●  modelo pronto  •  imgsz auto 640 / 320",
                 bg="#052e16", fg="#86efac", font=("Segoe UI", 9)).pack(padx=10, pady=3)

        # ---- cartões ----
        meio = tk.Frame(root, bg=BG)
        meio.pack(fill="x", padx=24, pady=14)

        card_cam = self._cartao(
            meio, icone="📷", titulo="Câmera ao vivo",
            desc="Webcam com detecção em tempo real\n+ pausa/retoma + salvar em pasta",
            btn_texto="Ligar câmera", cor=VERDE, cor_hover=VERDE_HOVER,
            comando=self.abrir_webcam)
        card_cam.pack(side="left", expand=True, fill="both", padx=(0, 8))

        card_arq = self._cartao(
            meio, icone="📁", titulo="Arquivo",
            desc="Imagem ou vídeo do disco\nresultado anotado salvo em outputs/",
            btn_texto="Escolher arquivo", cor=AZUL, cor_hover=AZUL_HOVER,
            comando=self.escolher_arquivo)
        card_arq.pack(side="left", expand=True, fill="both", padx=(8, 0))

        dica = tk.Frame(root, bg=BG)
        dica.pack(fill="x", padx=24)
        tk.Label(dica, text=f"Resultados salvos em:  {OUTPUTS_DIR}",
                 bg=BG, fg="#64748b", font=("Segoe UI", 8)).pack(anchor="w")

        rodape = tk.Frame(root, bg="#020617", highlightbackground=CARD_BORDA,
                          highlightthickness=1)
        rodape.pack(side="bottom", fill="x")
        self.dot = tk.Label(rodape, text="●", bg="#020617", fg=VERDE,
                            font=("Segoe UI", 10))
        self.dot.pack(side="left", padx=(14, 4), pady=8)
        self.status = tk.Label(rodape, text="Pronto.", bg="#020617", fg=SUAVE,
                               font=("Segoe UI", 9))
        self.status.pack(side="left", pady=8)
        tk.Button(rodape, text="Sair", bg="#020617", fg=SUAVE, relief="flat",
                  activebackground="#020617", activeforeground=TEXTO,
                  font=("Segoe UI", 9, "bold"), cursor="hand2",
                  command=root.destroy).pack(side="right", padx=14)

        self.cap = None
        self.rodando = False
        self._falhas = 0
        self._pendente = None
        self.tempo_real_pausado = False
        self.win_cam = None
        self.lbl_cam = None
        self.lbl_info = None
        self.btn_pausar = None
        self._ultimo_anotado = None

    # ---------------- componentes ----------------
    def _cartao(self, pai, icone, titulo, desc, btn_texto, cor, cor_hover, comando):
        card = tk.Frame(pai, bg=CARD, highlightbackground=CARD_BORDA,
                        highlightthickness=1, padx=18, pady=16)
        tk.Label(card, text=icone, bg=CARD, fg=TEXTO,
                 font=("Segoe UI", 28)).pack(pady=(2, 4))
        tk.Label(card, text=titulo, bg=CARD, fg=TEXTO,
                 font=FONTE_CARD_TITULO).pack()
        tk.Label(card, text=desc, bg=CARD, fg=SUAVE,
                 font=FONTE_CARD_DESC, justify="center").pack(pady=(4, 12))
        btn = tk.Label(card, text=f"  {btn_texto}  ", bg=cor, fg="white",
                       font=("Segoe UI", 10, "bold"), padx=10, pady=8,
                       cursor="hand2")
        btn.pack(fill="x")
        btn.bind("<Button-1>", lambda e: comando())
        btn.bind("<Enter>", lambda e: btn.config(bg=cor_hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=cor))
        return card

    # ---------------- helpers thread-safe ----------------
    def _set_status(self, texto, ok=True):
        def _apl():
            self.status.config(text=texto)
            self.dot.config(fg=VERDE if ok else "#ef4444")
        self.root.after(0, _apl)

    def _erro(self, titulo, msg):
        self.root.after(0, lambda: messagebox.showerror(titulo, msg))

    def _info(self, titulo, msg):
        self.root.after(0, lambda: messagebox.showinfo(titulo, msg))

    # ---------------- Arquivo -> Explorador ----------------
    def escolher_arquivo(self):
        caminho = filedialog.askopenfilename(
            title="Escolher imagem ou vídeo",
            filetypes=[("Imagens/Vídeos", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                       ("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Vídeos", "*.mp4 *.avi *.mov *.mkv"),
                       ("Todos", "*.*")])
        if not caminho:
            return
        ext = Path(caminho).suffix.lower()
        self.status.config(text=f"Analisando {Path(caminho).name} ...")
        self.dot.config(fg="#eab308")
        threading.Thread(target=self._processar_arquivo,
                         args=(caminho, ext), daemon=True).start()

    def _processar_arquivo(self, caminho, ext):
        try:
            if ext in EXT_VIDEO:
                self._processar_video(caminho)
                return
            img_cv = cv2.imread(caminho)
            if img_cv is None:
                raise RuntimeError("Não foi possível ler a imagem.")
            h, w = img_cv.shape[:2]
            print(f"Imagem original {w}x{h} -> imgsz={IMGSZ_PADRAO_HF} automático")
            results = prever(caminho, imgsz=IMGSZ_PADRAO_HF, modo="imagem")
            cont = contar(results)
            anotada_bgr = results[0].plot()
            p = Path(caminho)
            saida = str(OUTPUTS_DIR / f"{p.stem}_detectado{p.suffix}")
            cv2.imwrite(saida, anotada_bgr)
            texto = f"{p.name} -> {cont or 'nada'}\nSalvo: {saida}"
            self.root.after(0, lambda: self._mostrar_resultado(anotada_bgr, texto))
            self._set_status(f"OK: {cont or 'nada detectado'} | Salvo em {saida}")
        except Exception as e:
            self._erro("Erro", str(e))
            self._set_status("Erro ao analisar arquivo.", ok=False)

    def _processar_video(self, caminho):
        try:
            self._set_status("Processando vídeo, aguarde ...")
            self.root.after(0, lambda: self.dot.config(fg="#eab308"))
            path, resumo, _ = processar_video(caminho)
            self._set_status(f"Vídeo: {resumo}")
            self._info("Vídeo processado", f"{resumo}\nArquivo: {path}")
            if path:
                self._tocar_video(path)
        except Exception as e:
            self._erro("Erro no vídeo", str(e))
            self._set_status("Erro ao processar vídeo.", ok=False)

    def _tocar_video(self, path):
        cap = cv2.VideoCapture(path)
        cv2.namedWindow("Video anotado (q fecha)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video anotado (q fecha)", 1280, 720)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video anotado (q fecha)", ajustar_para_tela(frame))
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _mostrar_resultado(self, anotada_bgr, titulo="Resultado"):
        img = Image.fromarray(ajustar_para_tela(anotada_bgr, 1280, 720)[:, :, ::-1])
        win = tk.Toplevel(self.root)
        win.title(titulo[:80])
        win.configure(bg=BG)
        foto = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=foto, bg=BG, highlightthickness=0)
        lbl.image = foto
        lbl.pack(padx=12, pady=(12, 4))
        tk.Label(win, text=titulo, bg=BG, fg=TEXTO,
                 font=("Segoe UI", 10)).pack(pady=6)

    # ---------------- Webcam tempo real ----------------
    def _tentar_abrir(self, indice):
        for backend in (0, getattr(cv2, "CAP_DSHOW", 0) or 0):
            cap = cv2.VideoCapture(indice, backend) if backend else cv2.VideoCapture(indice)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        return None

    def abrir_webcam(self):
        if self.win_cam is not None and self.win_cam.winfo_exists():
            self.win_cam.lift()
            return
        cap = self._tentar_abrir(0)
        if cap is None:
            messagebox.showerror(
                "Webcam",
                "Não foi possível abrir a câmera.\n\n"
                "Verifique:\n"
                "• Outro app usando a webcam (Meet, Teams, Câmera)?\n"
                "• Windows: Configurações > Privacidade e segurança > Câmera\n"
                "  > permitir o acesso p/ 'aplicativos de área de trabalho'.\n"
                "• PC sem webcam? Use 'Escolher arquivo'.")
            self.status.config(text="Webcam indisponível.")
            self.dot.config(fg="#ef4444")
            return
        self.cap = cap
        self.rodando = True
        self.tempo_real_pausado = False
        self._falhas = 0
        self._pendente = None
        self.win_cam = tk.Toplevel(self.root)
        self.win_cam.title("Webcam ao vivo — YOLO26n (feche p/ parar)")
        self.win_cam.configure(bg=BG)
        self.win_cam.minsize(680, 560)
        self.win_cam.protocol("WM_DELETE_WINDOW", self.fechar_webcam)

        self.lbl_cam = tk.Label(self.win_cam, text="Abrindo câmera…",
                                bg=BG, fg=SUAVE, font=("Segoe UI", 11),
                                width=80, height=20)
        self.lbl_cam.pack(padx=10, pady=(10, 2))

        self.lbl_info = tk.Label(self.win_cam, text="...", bg=BG, fg=VERDE,
                                 font=("Consolas", 10))
        self.lbl_info.pack(pady=4)

        frm = tk.Frame(self.win_cam, bg=BG)
        frm.pack(pady=8)

        # Botões da webcam
        botoes = [
            ("💾 Salvar frame", self.salvar_frame, AZUL),
            ("⏸ Pausar tempo real", self.alternar_pausa, LARANJA),
            ("⏹ Parar", self.fechar_webcam, "#ef4444"),
        ]
        for texto, cmd, cor in botoes:
            b = tk.Label(frm, text=f"  {texto}  ", bg=cor, fg="white",
                         font=("Segoe UI", 10, "bold"), padx=10, pady=6,
                         cursor="hand2")
            b.pack(side="left", padx=8)
            b.bind("<Button-1>", lambda e, c=cmd: c())
            if "Pausar" in texto:
                self.btn_pausar = b

        self.status.config(text="Webcam ligada — tempo real ATIVO")
        self.dot.config(fg="#eab308")
        threading.Thread(target=self._worker_webcam, args=(cap,), daemon=True).start()

    def _worker_webcam(self, cap):
        primeira = True
        while self.rodando:
            ret, frame = cap.read()
            if not ret or frame is None:
                self._falhas += 1
                if self._falhas > 30:
                    self.root.after(0, self._erro_leitura)
                    break
                time.sleep(0.05)
                continue
            self._falhas = 0
            frame = cv2.flip(frame, 1)
            if primeira:
                primeira = False
                self.root.after(0, lambda: self.lbl_info.config(
                    text="Carregando modelo (1ª vez demora)…"))

            if not self.tempo_real_pausado:
                t0 = time.time()
                try:
                    results = prever(frame, imgsz=IMGSZ_STREAM_PADRAO, modo="stream")
                except Exception as e:
                    self.root.after(0, lambda e=e: self._erro("Erro na detecção", str(e)))
                    break
                dt = (time.time() - t0) * 1000
                cont = contar(results)
                anotada = results[0].plot()
                self._ultimo_anotado = anotada
                fps = 1000.0 / max(dt, 1e-6)
                h, w = frame.shape[:2]
                info = f"{w}x{h} -> {IMGSZ_STREAM_PADRAO} | {dt:.0f}ms {fps:.1f}FPS | {cont or 'nada'}"
                self._pendente = (ajustar_para_tela(anotada, 960, 540), info)
            else:
                # Pausado: mostra frame original sem detecção
                info = "⏸ TEMPO REAL PAUSADO"
                self._pendente = (ajustar_para_tela(frame, 960, 540), info)

            self.root.after(0, self._pintar)

    def _pintar(self):
        if (self._pendente is None or self.lbl_cam is None
                or self.win_cam is None or not self.win_cam.winfo_exists()):
            return
        pequeno, info = self._pendente
        self._pendente = None
        self.lbl_info.config(text=info)
        foto = ImageTk.PhotoImage(Image.fromarray(pequeno[:, :, ::-1]))
        self.lbl_cam.configure(image=foto, text="", width=0, height=0)
        self.lbl_cam.image = foto

    def _erro_leitura(self):
        messagebox.showerror("Webcam", "A câmera parou de enviar imagens.")
        self.fechar_webcam()

    def alternar_pausa(self):
        self.tempo_real_pausado = not self.tempo_real_pausado
        if self.tempo_real_pausado:
            self.btn_pausar.config(text="▶ Retomar tempo real", bg=VERDE)
            self._set_status("⏸ Tempo real PAUSADO")
        else:
            self.btn_pausar.config(text="⏸ Pausar tempo real", bg=LARANJA)
            self._set_status("▶ Tempo real ATIVO")

    def salvar_frame(self):
        if self._ultimo_anotado is None:
            return
        # Permite escolher pasta
        pasta = filedialog.askdirectory(title="Escolher pasta para salvar", initialdir=str(OUTPUTS_DIR))
        if not pasta:
            return
        nome = Path(pasta) / f"webcam_{int(time.time())}.jpg"
        cv2.imwrite(str(nome), self._ultimo_anotado)
        messagebox.showinfo("Salvo", str(nome))

    def fechar_webcam(self):
        self.rodando = False
        time.sleep(0.05)
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self.win_cam is not None and self.win_cam.winfo_exists():
            self.win_cam.destroy()
        self.win_cam = None
        self._ultimo_anotado = None
        self.btn_pausar = None
        self.status.config(text="Webcam desligada.")
        self.dot.config(fg=VERDE)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()