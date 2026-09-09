"""Qotgan (harakatsiz) topilmalarni sanoqdan chiqaradi.

Muammo: kamera devordagi RASM/MUROL/PLAKATni (odam surati) odam deb sanaydi
(o'lchandi: Nashville jonli oqimida devordagi murol qutisining vaqt bo'yicha
piksel std = 0.00 — mutlaqo qotgan). Real odam esa — hatto o'tirgani ham —
qimirlaydi, nafas oladi, yorug'lik/soya o'zgaradi: uning std'i nolga teng
bo'lmaydi.

Yechim: har kamera uchun oxirgi bir nechta kadr saqlanadi; topilma qutisi
ichidagi piksellar vaqt bo'yicha deyarli o'zgarmasa (std < STATIC_STD),
u RASM deb belgilanadi va sanalmaydi.

Ehtiyot: chegara PAST (1.0) — faqat haqiqatan qotgan (rasm/plakat) chiqadi,
o'tirgan tirik odam emas.
"""
import threading
from collections import deque

import cv2
import numpy as np

BUF = 6            # nechta oxirgi kadr saqlanadi
MIN_FRAMES = 4     # baho berish uchun kamida shuncha kadr kerak
STATIC_STD = 1.0   # region vaqt-std shundan past = qotgan (rasm/murol/plakat)
SCALE = 0.5        # kadrni yarim o'lchamda saqlaymiz (xotira/tezlik)

_bufs = {}
_lock = threading.Lock()


def _buf(key):
    with _lock:
        d = _bufs.get(key)
        if d is None:
            d = _bufs[key] = deque(maxlen=BUF)
        return d


def update_and_filter(key, frame, persons):
    """Kadrni tarixга qo'shadi va qotgan topilmalarni olib tashlaydi.

    Qaytaradi: faqat harakatli (tirik) topilmalar. Qotganlarга p["static"]=True
    qo'yiladi (kerak bo'lsa ko'rsatish uchun), lekin ro'yxatдан chiqariladi.
    """
    d = _buf(key)
    small = cv2.cvtColor(cv2.resize(frame, None, fx=SCALE, fy=SCALE),
                         cv2.COLOR_BGR2GRAY)
    d.append(small)
    if len(d) < MIN_FRAMES:
        return persons          # hali yetarli tarix yo'q — hammasini qoldiramiz
    stack = np.stack(list(d)[-MIN_FRAMES:]).astype(np.float32)
    H, W = small.shape
    out = []
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        rx1, ry1 = max(0, int(x1 * SCALE)), max(0, int(y1 * SCALE))
        rx2, ry2 = min(W, int(x2 * SCALE)), min(H, int(y2 * SCALE))
        if rx2 <= rx1 or ry2 <= ry1:
            out.append(p)
            continue
        reg = stack[:, ry1:ry2, rx1:rx2]
        std = float(reg.std(axis=0).mean())
        if std < STATIC_STD:
            p["static"] = True   # qotgan — rasm/murol/plakat, sanamaymiz
            continue
        out.append(p)
    return out
