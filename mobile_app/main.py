"""Cliente Android (Kivy): fotografa/escolhe no celular e envia ao PC (app_web.py).

Build do APK no Linux/Colab:  buildozer android debug
Pré-requisito: PC rodando  py app_web.py  no mesmo Wi-Fi.
"""
import threading
from pathlib import Path

import requests
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput


class Tela(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=12, spacing=8, **kw)
        self.url = TextInput(text="http://192.168.0.10:5000/api/detect",
                             size_hint_y=None, height=44, multiline=False)
        self.add_widget(Label(text="Servidor (IP do PC):", size_hint_y=None,
                              height=24, color=(0.58, 0.64, 0.71, 1)))
        self.add_widget(self.url)
        linha = BoxLayout(size_hint_y=None, height=52, spacing=8)
        b_cam = Button(text="📷 Fotografar", background_color=(0.13, 0.77, 0.37, 1))
        b_gal = Button(text="📁 Galeria", background_color=(0.23, 0.51, 0.96, 1))
        b_cam.bind(on_release=self.fotografar)
        b_gal.bind(on_release=self.galeria)
        linha.add_widget(b_cam)
        linha.add_widget(b_gal)
        self.add_widget(linha)
        self.status = Label(text="Aponte o servidor e envie uma foto.",
                            size_hint_y=None, height=30)
        self.add_widget(self.status)
        self.img = AsyncImage(source="", allow_stretch=True, keep_ratio=True)
        self.add_widget(self.img)

    def _set(self, texto):
        Clock.schedule_once(lambda dt: setattr(self.status, "text", texto))

    def fotografar(self, *_):
        try:
            from plyer import camera
            dst = str(Path(App.get_running_app().user_data_dir) / "foto.jpg")
            camera.take_picture(filename=dst, on_complete=lambda p: self.enviar(p or dst))
        except Exception as e:
            self._set(f"Câmera indisponível: {e}")

    def galeria(self, *_):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=lambda s: s and self.enviar(s[0]),
                                  filters=[("Imagens", "*.jpg", "*.jpeg", "*.png", "*.webp")])
        except Exception as e:
            self._set(f"Galeria indisponível: {e}")

    def enviar(self, caminho):
        self._set("Enviando…")
        threading.Thread(target=self._post, args=(caminho,), daemon=True).start()

    def _post(self, caminho):
        try:
            with open(caminho, "rb") as fh:
                r = requests.post(self.url.text.strip(), files={"imagem": fh},
                                  timeout=60)
            j = r.json()
            if not j.get("ok"):
                raise RuntimeError(j.get("erro", "falha"))
            import base64, tempfile
            tmp = tempfile.mktemp(suffix=".jpg")
            with open(tmp, "wb") as fh:
                fh.write(base64.b64decode(j["imagem_b64"]))
            Clock.schedule_once(lambda dt: setattr(self.img, "source", tmp))
            self._set(f'{j["tempo_ms"]} ms • {j["resumo"]}')
            ok = Popup(title="Resultado", content=Label(text=j["resumo"]),
                       size_hint=(0.8, 0.4))
            Clock.schedule_once(lambda dt: ok.open())
        except Exception as e:
            self._set(f"Erro: {e}")


class DetectorApp(App):
    title = "YOLO26n"
    icon = "../assets/icon.png"

    def build(self):
        return Tela()


if __name__ == "__main__":
    DetectorApp().run()
