"""Yuz tanish — kadrdagi odam kim ekanini aniqlaydi.

Qoidalarning ko'pi "kim" degan savolga tayanadi: 3.1 (mentor darsni boshladimi),
3.3 (xonadan chiqib ketdimi), 2.1 (kechikdimi). Anonim "odam bor" yetarli emas —
jarima aniq xodimga qo'yiladi.

Modellar OpenCV bilan keladi, tashqi API yo'q:
  YuNet  — yuzni topadi (models/yunet.onnx)
  SFace  — 128 o'lchovli embedding beradi (models/sface.onnx)

Baza: data/faces.json {ism: [embedding, ...]}, data/meta.json {ism: {filiallar}}.
voice-face-id loyihasidan ko'chirildi — ovoz (ECAPA) qismi olinmadi.

Mustaqil sinash:
    ./venv/bin/python faces.py rasm.jpg
"""
import os
import json
import threading

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "models")
DATA_DIR = os.path.join(_HERE, "data")
FACE_DB = os.path.join(DATA_DIR, "faces.json")
META_DB = os.path.join(DATA_DIR, "meta.json")

# SFace kosinus chegarasi. OpenCV ~0.36 tavsiya qiladi; 0.40 — biroz qattiqroq,
# chunki noto'g'ri odamga jarima qo'yish yo'q signaldan qimmatroq.
THRESHOLD = 0.40
DET_SIZE = (320, 320)

_det = _rec = None
_lock = threading.Lock()


def _models():
    global _det, _rec
    with _lock:
        if _det is None:
            _det = cv2.FaceDetectorYN_create(
                os.path.join(MODEL_DIR, "yunet.onnx"), "", DET_SIZE, 0.5)
            _rec = cv2.FaceRecognizerSF_create(
                os.path.join(MODEL_DIR, "sface.onnx"), "")
        return _det, _rec


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# Baza kadrma-kadr o'qilmasin — sekin. Fayl o'zgarsa qayta o'qiladi.
_db_cache = {"mtime": None, "known": {}}


def known_faces(branch=None):
    """{ism: [normallashtirilgan embedding, ...]}. branch berilsa — o'sha filial."""
    mtime = os.path.getmtime(FACE_DB) if os.path.exists(FACE_DB) else None
    if _db_cache["mtime"] != mtime:
        db = _load(FACE_DB)
        _db_cache.update(
            mtime=mtime,
            known={name: [_unit(e) for e in embs] for name, embs in db.items()})
    known = _db_cache["known"]
    if not branch:
        return known
    meta = _load(META_DB)
    filtered = {n: e for n, e in known.items()
                if branch in meta.get(n, {}).get("filiallar", [])}
    # Filialga hech kim biriktirilmagan bo'lsa hammasi bilan solishtiramiz —
    # aks holda yangi filialda kamera hech kimni tanimay qoladi.
    return filtered or known


def detect(frame):
    """Kadrdagi yuzlar: [(x, y, w, h, ishonch), ...]"""
    det, _ = _models()
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces]


def identify(frame, branch=None):
    """Kadrdagi har bir yuzni tanib beradi.

    Qaytaradi: [{box: (x,y,w,h), name: str|None, score: float}, ...]
    name=None — baza bilan mos kelmadi (begona yoki sifat past).
    """
    det, rec = _models()
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None:
        return []

    known = known_faces(branch)
    out = []
    for f in faces:
        aligned = rec.alignCrop(frame, f)
        emb = _unit(rec.feature(aligned).flatten())
        best, score = None, -1.0
        for name, embs in known.items():
            s = max(float(np.dot(e, emb)) for e in embs)
            if s > score:
                best, score = name, s
        out.append({
            "box": (int(f[0]), int(f[1]), int(f[2]), int(f[3])),
            "name": best if score >= THRESHOLD else None,
            "score": round(score, 3),
        })
    return out


def annotate(frame, people):
    """identify() natijasini kadrga chizadi (joyida o'zgartiradi)."""
    for p in people:
        x, y, w, h = p["box"]
        named = p["name"] is not None
        color = (0, 200, 0) if named else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{p['name']} {p['score']:.2f}" if named else f"? {p['score']:.2f}"
        cv2.putText(frame, label, (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


if __name__ == "__main__":
    import sys
    known = known_faces()
    print(f"Bazada {len(known)} ta xodim")
    if len(sys.argv) < 2:
        for name in sorted(known):
            print(f"  {name} — {len(known[name])} ta embedding")
        raise SystemExit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"rasm o'qilmadi: {sys.argv[1]}")
    people = identify(img)
    print(f"{len(people)} ta yuz topildi:")
    for p in people:
        print(f"  {p['name'] or '(tanilmadi)':<25} {p['score']}  {p['box']}")
