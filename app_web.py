"""Servidor web mobile (PWA): rode no PC e acesse pelo celular no mesmo Wi-Fi.

    py app_web.py [--port 5000]

No celular: abra http://<IP-DO-PC>:5000  ->  "Adicionar à tela inicial"
para instalar como app (ícone próprio, tela cheia).
"""
import argparse
import base64
import io
import socket
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.inference import anotar_bgr, contar, prever, resumir_contagem

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

HTML = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0f172a">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon-192.png">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>YOLO26n — Detector</title>
<style>
*{box-sizing:border-box;margin:0}body{background:#0f172a;color:#f1f5f9;
font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;min-height:100dvh}
header{padding:20px 18px 10px}header .eyebrow{color:#22c55e;font-weight:700;
font-size:12px;letter-spacing:.08em}h1{font-size:26px;margin:2px 0}
.sub{color:#94a3b8;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:14px 18px}
.card{background:#1e293b;border:1px solid #334155;border-radius:14px;
padding:16px 12px;text-align:center}
.card .ic{font-size:34px}.card h2{font-size:15px;margin:6px 0 2px}
.card p{color:#94a3b8;font-size:11px;margin-bottom:10px}
.btn{display:block;border-radius:10px;padding:12px;font-weight:700;
font-size:14px;color:#fff;text-decoration:none}
.verde{background:#22c55e}.azul{background:#3b82f6}
#prev,#res{width:100%;border-radius:12px;display:none;margin-top:10px}
main{padding:0 18px 30px}.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{background:#052e16;color:#86efac;border:1px solid #166534;
border-radius:20px;padding:4px 12px;font-size:12px}
#status{color:#94a3b8;font-size:13px;text-align:center;padding:8px}
.spin{animation:g 1s linear infinite;display:inline-block}
@keyframes g{to{transform:rotate(360deg)}}
footer{text-align:center;color:#64748b;font-size:11px;padding:0 0 20px}
</style></head><body>
<header><div class="eyebrow">◉ DETECTOR DE OBJETOS</div>
<h1>YOLO26n</h1><div class="sub">Ultralytics / YOLO26 • modelo nano</div></header>
<div class="grid">
<div class="card"><div class="ic">📷</div><h2>Câmera</h2>
<p>Foto na hora</p>
<label class="btn verde" for="cam">Fotografar</label>
<input id="cam" type="file" accept="image/*" capture="environment" hidden></div>
<div class="card"><div class="ic">📁</div><h2>Galeria</h2>
<p>Imagem salva</p>
<label class="btn azul" for="arq">Escolher</label>
<input id="arq" type="file" accept="image/*" hidden></div>
</div>
<main><div id="status">Escolha uma opção acima.</div>
<img id="prev" alt="prévia"><img id="res" alt="resultado"><div id="chips" class="chips"></div></main>
<footer>Servidor local • resultados também salvos em outputs/ no PC</footer>
<script>
const st=document.getElementById('status'),ch=document.getElementById('chips'),
prev=document.getElementById('prev'),res=document.getElementById('res');
async function envia(f){prev.src=URL.createObjectURL(f);prev.style.display='block';
res.style.display='none';ch.innerHTML='';
st.innerHTML='<span class="spin">◌</span> Detectando…';
const fd=new FormData();fd.append('imagem',f);
try{const r=await fetch('/api/detect',{method:'POST',body:fd});
const j=await r.json();if(!j.ok)throw new Error(j.erro||'falha');
res.src='data:image/jpeg;base64,'+j.imagem_b64;res.style.display='block';
st.textContent=j.tempo_ms+' ms • '+j.resumo;
ch.innerHTML=Object.entries(j.contagem).map(([k,v])=>
`<span class="chip">${k}: ${v}</span>`).join('')||'<span class="chip">nada detectado</span>';
}catch(e){st.textContent='Erro: '+e.message;}}
cam.onchange=e=>e.target.files[0]&&envia(e.target.files[0]);
arq.onchange=e=>e.target.files[0]&&envia(e.target.files[0]);
if('serviceWorker' in navigator)navigator.serviceWorker.register('/sw.js').catch(()=>{});
</script></body></html>"""

MANIFEST = {
    "name": "YOLO26n — Detector de Objetos",
    "short_name": "YOLO26n",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0f172a",
    "theme_color": "#0f172a",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
    ],
}

SW = """self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('fetch',e=>{});"""


@app.get("/")
def index():
    return HTML


@app.get("/manifest.json")
def manifest():
    return jsonify(MANIFEST)


@app.get("/sw.js")
def sw():
    return app.response_class(SW, mimetype="application/javascript")


@app.get("/icon-192.png")
def icon192():
    return send_from_directory(BASE / "assets", "icon-192.png")


@app.get("/icon-512.png")
def icon512():
    return send_from_directory(BASE / "assets", "icon-512.png")


@app.post("/api/detect")
def api_detect():
    f = request.files.get("imagem")
    if f is None:
        return jsonify(ok=False, erro="nenhuma imagem enviada"), 400
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, erro="imagem inválida"), 400
    t0 = time.time()
    results = prever(img, modo="imagem")  # imgsz AUTO (640 p/ imagem)
    dt_ms = round((time.time() - t0) * 1000, 1)
    cont = contar(results)
    _, jpg = cv2.imencode(".jpg", anotar_bgr(results),
                          [cv2.IMWRITE_JPEG_QUALITY, 88])
    return jsonify(ok=True, tempo_ms=dt_ms, contagem=cont,
                   resumo=resumir_contagem(cont),
                   imagem_b64=base64.b64encode(jpg.tobytes()).decode())


def ips_locais():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ip.startswith(("192.168.", "10.", "172.")):
                ips.add(ip)
    except Exception:
        pass
    if not ips:  # fallback: força descoberta via socket de saída
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return sorted(ips) or ["127.0.0.1"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Servidor web mobile YOLO26n (PWA)")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)
    print("No celular (mesmo Wi-Fi), abra:")
    for ip in ips_locais():
        print(f"  ->  http://{ip}:{args.port}")
    print("Depois: menu do navegador > 'Adicionar à tela inicial'.")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
