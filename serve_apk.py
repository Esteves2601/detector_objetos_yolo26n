import http.server, socketserver, socket, qrcode, os, threading, subprocess

os.chdir(r'C:\Users\Estevão\PycharmProjects\detector_objetos\android\app\build\outputs\apk\debug')

# Get local IP
out = subprocess.check_output(['ipconfig'], text=True, encoding='cp850', errors='ignore')
ip = '127.0.0.1'
for line in out.splitlines():
    if 'IPv4' in line:
        cand = line.split(':')[-1].strip()
        if cand.startswith(('192.168.','10.','172.')):
            ip = cand
            break

port = 8080
url = f'http://{ip}:{port}/app-debug.apk'
img = qrcode.make(url)
img.save('qr_apk.png')
print(f'APK em: {url}')

server = socketserver.TCPServer(('', port), http.server.SimpleHTTPRequestHandler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()

print('Servidor rodando. Ctrl+C para parar.')
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.shutdown()