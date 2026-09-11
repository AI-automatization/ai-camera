"""Tana (ko'rinish) vektori — person re-identification (boxmot ReID).

Yuz ko'rinmasa ham odamni O'SHA KUN ichida tanib qolish uchun: kiyim, qomat,
soch, turish — butun odam qutisi 512 songa aylanadi (ArcFace yuzga qilgani kabi).

O'lchandi (2026-09-10, A5: hamma oq ko'ylakda, shift burchagi, kichik qutilar):
  model                 bir odam(2 soat farq)  boshqa odam(max)  ms/odam
  osnet_x0_25            0.71                   0.70              6      ← ajratmaydi
  osnet_x1_0             0.72                   0.54              12
  clip_market1501        0.74                   0.50              32     ← eng yaxshi (default)
  lmbn_n_duke            0.68                   0.46              25
Bir kadr ichidagi ikki odam 0.59 gacha o'xshaydi — ularni "bir vaqtda ekranda"
qoidasi ajratadi (identity.py), vektor emas.

Tana vektori YOLG'IZ ishonchli emas (bir xil kiyim) — faqat yuz bilan birga
va qoidalar ostida ishlatiladi. REID_MODEL env bilan model almashtiriladi.
"""
import os
import threading
import time

import numpy as np

MODEL = os.environ.get("REID_MODEL", "clip_market1501.pt")
MIN_H = int(os.environ.get("BODY_MIN_H", "110"))     # bundan past quti — vektor olinmaydi
EDGE = 0.01                                          # kadr chetiga tegib kesilgan quti — vektor yo'q
_reid = None
_lock = threading.Lock()


def load_models():
    return _model()


def _model():
    global _reid
    with _lock:
        if _reid is None:
            t0 = time.time()
            from boxmot.reid.core.reid import ReID
            from boxmot.utils import WEIGHTS
            _reid = ReID(weights=WEIGHTS / MODEL, device="cpu")
            print(f"[body] ReID {MODEL} yuklandi ({time.time() - t0:.1f}s)")
        return _reid


def embed(frame, persons):
    """persons ro'yxati uchun normalangan vektorlar (yoki None — quti kichik).

    Qaytaradi: [emb | None, ...] persons tartibida.
    """
    H, W = frame.shape[:2]
    mx, my = W * EDGE, H * EDGE

    def whole(b):
        x1, y1, x2, y2 = b
        return x1 > mx and y1 > my and x2 < W - mx and y2 < H - my

    idx = [i for i, p in enumerate(persons)
           if (p["box"][3] - p["box"][1]) >= MIN_H and whole(p["box"])]
    out = [None] * len(persons)
    if not idx:
        return out
    boxes = np.array([persons[i]["box"] for i in idx], np.float32)
    feats = np.asarray(_model()(frame, boxes), np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    for k, i in enumerate(idx):
        out[i] = feats[k]
    return out


def sim(a, b):
    return float(np.dot(a, b))


def geometry(p):
    """Qomat belgisi: yelka kengligi / quti balandligi (bo'g'imlar bo'lsa).

    Yakka ishlatilmaydi — faqat VETO: bir xil kiyimda farq katta bo'lsa
    "boshqa odam". Kamera burchagiga bog'liq, shuning uchun bag'rikeng chegara.
    """
    k = p.get("keypoints")
    if not k:
        return None
    try:
        ls, rs = k[5], k[6]           # COCO: 5 chap yelka, 6 o'ng yelka
        if ls[2] < 0.3 or rs[2] < 0.3:
            return None
        sw = abs(float(ls[0]) - float(rs[0]))
        h = float(p["box"][3] - p["box"][1])
        if h <= 0 or sw <= 0:
            return None
        return {"shoulder_ratio": round(sw / h, 3)}
    except (IndexError, TypeError, KeyError):
        return None


def geometry_conflict(g1, g2, tol=0.40):
    """Ikki qomat belgisi qarama-qarshimi (nisbiy farq > tol)."""
    if not g1 or not g2:
        return False
    a, b = g1.get("shoulder_ratio"), g2.get("shoulder_ratio")
    if not a or not b:
        return False
    return abs(a - b) / max(a, b) > tol
