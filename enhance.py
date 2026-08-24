"""Kadrni ko'zga tiniqroq qilish — FAQAT ko'rsatish uchun.

MUHIM: bu modelga berilmaydi. O'lchandi (B1, 2026-08-24): o'tkirlash
aniqlashni yomonlashtiradi — 5 odam topilgan kadrda kuchli o'tkirlashdan
keyin 3 tasi qoldi. Model tabiiy tasvirda o'qitilgan, sun'iy o'tkirlangan
kadr unga notanish ko'rinadi.

Shuning uchun ikki yo'l ajratilgan:
    model  -> asl kadr
    ekran  -> shu yerdagi ishlov

Dars xonalari qorong'i va tasvir yumshoq (o'lchangan tiniqlik: B4=808,
B1=891, B2=964), koridorlar esa tiniq (Oshxona=4223, Admin 2=4840).
Sabab kamera sozlamasida emas — ikkalasida ham bir xil (VBR, 1024 kbps).
Farq sahnada: qorong'ida kamera gain ni ko'taradi, tasvir yumshaydi.
"""
import cv2
import numpy as np

CLAHE_CLIP = 2.0        # mahalliy kontrast; kattaroq qilinsa shovqin chiqadi
CLAHE_GRID = (8, 8)
SHARPEN = 0.55          # yumshoq o'tkirlash (1.0 da halo paydo bo'ladi)
SHARPEN_RADIUS = 1.2

_clahe = None


def _get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return _clahe


def enhance(frame):
    """Kontrastni tekislaydi va yumshoq o'tkirlaydi.

    LAB fazasida faqat yorug'lik kanali (L) ishlanadi — RGB da qilinsa
    ranglar buziladi (sinovda devor va teri rangi siljigan edi).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _get_clahe().apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), SHARPEN_RADIUS)
    return cv2.addWeighted(out, 1 + SHARPEN, blur, -SHARPEN, 0)


def enhance_jpeg(data, quality=92):
    """JPEG baytlarni ishlab, yana JPEG qaytaradi. Xato bo'lsa aslini."""
    try:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return data
        ok, buf = cv2.imencode(".jpg", enhance(frame),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else data
    except Exception:
        return data


def sharpness(frame):
    """Tiniqlik o'lchovi (Laplacian variansi). Kattasi — tiniqroq."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())
