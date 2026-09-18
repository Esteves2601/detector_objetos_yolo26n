"""Gera ícone criativo do app: olho-detector + bounding box YOLO."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(exist_ok=True)

BG0, BG1 = (15, 23, 42), (30, 41, 59)   # navy -> slate
VERDE = (34, 197, 94)
BRANCO = (241, 245, 249)
AMARELO = (234, 179, 8)


def gradiente(n):
    base = Image.new("RGB", (n, n), BG0)
    px = base.load()
    for y in range(n):
        t = y / (n - 1)
        c = tuple(int(a + (b - a) * t) for a, b in zip(BG0, BG1))
        for x in range(n):
            px[x, y] = c
    # cantos arredondados
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n, n], radius=n // 6, fill=255)
    base.putalpha(mask)
    return base


def desenhar(n=512):
    img = gradiente(n).convert("RGBA")
    d = ImageDraw.Draw(img)
    m = n // 8
    lw = max(6, n // 64)
    # bounding box principal (verde, cantos em destaque)
    d.rounded_rectangle([m, m, n - m, n - m], radius=n // 16,
                        outline=VERDE + (255,), width=lw)
    c = n // 14  # tamanho dos cantos
    for x0, y0, dx, dy in ((m, m, 1, 1), (n - m, m, -1, 1),
                           (m, n - m, 1, -1), (n - m, n - m, -1, -1)):
        d.line([x0, y0, x0 + dx * c, y0], fill=AMARELO + (255,), width=lw + 4)
        d.line([x0, y0, x0, y0 + dy * c], fill=AMARELO + (255,), width=lw + 4)
    # olho central (câmera que detecta)
    cx = cy = n // 2
    r = n // 5
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BRANCO + (255,))
    rp = int(r * 0.52)
    d.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], fill=VERDE + (255,))
    rb = int(r * 0.22)
    d.ellipse([cx - rb, cy - rb, cx + rb, cy + rb], fill=(2, 6, 23, 255))
    # brilho
    rh = int(r * 0.12)
    d.ellipse([cx - rp // 2, cy - rp // 2 - rh, cx - rp // 2 + 2 * rh,
               cy - rp // 2 + rh], fill=BRANCO + (255,))
    # chip "26n"
    try:
        fnt = ImageFont.truetype("arial.ttf", n // 14)
    except Exception:
        fnt = ImageFont.load_default()
    txt = "YOLO·26n"
    bb = d.textbbox((0, 0), txt, font=fnt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = n // 40
    x0 = (n - tw) // 2
    y0 = n - m - th - pad * 3
    d.rounded_rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                        radius=n // 40, fill=(2, 6, 23, 230))
    d.text((x0, y0), txt, font=fnt, fill=VERDE + (255,))
    return img


if __name__ == "__main__":
    big = desenhar(512)
    big.save(OUT / "icon.png")
    big.resize((192, 192)).save(OUT / "icon-192.png")
    big.resize((512, 512)).save(OUT / "icon-512.png")
    big.save(OUT / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48),
                                      (64, 64), (128, 128), (256, 256)])
    print("OK:", sorted(p.name for p in OUT.iterdir()))
