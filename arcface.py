"""Yuz tanish — ArcFace (InsightFace buffalo_l) bilan.

SFace (faces.py) burchakka zaif edi: bir odam portret ↔ tepa-burchakда
0.12-0.20 ball berardi (o'lchandi). ArcFace millionlab yuzда o'qitilган,
burchakка ancha chidamli — boshqa odamlar ≈ 0.0, bir odam turli burcakда
ham yuqori qoladi.

Bir xil API (identify/enroll/enroll_sample/assess/people/remove/known_faces/
audit/load_models) — app.py o'zgarmaydi. Baza: data/arcface_db.json (512
o'lchov), meta va avatar faces.py bilan bo'lishiladi.

Detektor+embedding: InsightFace app.get() (SCRFD + ArcFace) — onnxruntime CPU,
MPS bilan to'qnashmaydi.
"""
import os
import threading

import cv2
import numpy as np

import faces as _f          # meta/avatar/_load/_save yordamchilari uchun

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
FACE_DB = os.path.join(DATA_DIR, "arcface_db.json")
META_DB = _f.META_DB

# ArcFace kosinus chegarasi. Boshqa odamlar ~0.05, bir odam (yaxshi) 0.4+.
# 0.30 — ehtiyotkor boshlang'ich; jonli sinovдан keyin sozlanadi.
THRESHOLD = 0.30
MIN_RECOGNIZE_PX = 40       # SCRFD mayda yuzни ham topadi, lekin juda mayда shovqin
MIN_ENROLL_PX = 55
ENROLL_TARGET = 14
ENROLL_NEW_MAX = 0.90       # yangi namuna mavjuddan shu qadar farq qilsin
ENROLL_SAME_MIN = 0.20      # lekin butunlay boshqa odam bo'lmasin
ENROLL_SHARP_MIN = 12

_app = None
_lock = threading.Lock()


def load_models():
    return _arc()


def _arc():
    global _app
    with _lock:
        if _app is None:
            from insightface.app import FaceAnalysis
            a = FaceAnalysis(name="buffalo_l",
                             providers=["CPUExecutionProvider"])
            a.prepare(ctx_id=-1, det_size=(640, 640))
            _app = a
        return _app


def _faces(frame):
    """Kadrdagi yuzlar (InsightFace Face obyektlari), kattadan kichikka."""
    fs = _arc().get(frame)
    fs.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True)
    return fs


def _box(f):
    x1, y1, x2, y2 = (int(v) for v in f.bbox)
    return (x1, y1, x2 - x1, y2 - y1)


def _sharp(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Baza ─────────────────────────────────────────────────────────────
_cache = {"mtime": None, "known": {}}


def known_faces(branch=None):
    mtime = os.path.getmtime(FACE_DB) if os.path.exists(FACE_DB) else None
    if _cache["mtime"] != mtime:
        db = _f._load(FACE_DB)
        _cache.update(mtime=mtime,
                      known={n: [np.asarray(e, np.float32) for e in v]
                             for n, v in db.items()})
    known = _cache["known"]
    if not branch:
        return known
    meta = _f._load(META_DB)
    filtered = {n: e for n, e in known.items()
                if branch in meta.get(n, {}).get("filiallar", [])}
    return filtered or known


def identify(frame, branch=None):
    known = known_faces(branch)
    out = []
    for f in _faces(frame):
        box = _box(f)
        if box[2] < MIN_RECOGNIZE_PX:
            out.append({"box": box, "name": None, "score": None,
                        "too_small": True})
            continue
        emb = f.normed_embedding
        best, score = None, -1.0
        for name, embs in known.items():
            s = max(float(np.dot(e, emb)) for e in embs)
            if s > score:
                best, score = name, s
        out.append({"box": box, "name": best if score >= THRESHOLD else None,
                    "score": round(score, 3), "too_small": False})
    return out


def assess(frame):
    fs = _faces(frame)
    if not fs:
        return {"ok": False, "reason": "Yuz ko'rinmayapti", "px": 0}
    f = fs[0]
    x, y, w, h = _box(f)
    if w < MIN_ENROLL_PX:
        return {"ok": False, "reason": f"Yaqinroq keling ({w}px)", "px": w}
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return {"ok": False, "reason": "Yuz kadr chetida", "px": w}
    sharp = _sharp(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    if sharp < ENROLL_SHARP_MIN:
        return {"ok": False, "reason": "Xira — sekinroq harakatlaning",
                "px": w, "sharp": round(sharp)}
    return {"ok": True, "reason": "", "px": w, "sharp": round(sharp),
            "emb": f.normed_embedding, "crop": crop}


def enroll_sample(frame, name, branches=None):
    name = (name or "").strip()
    if not name:
        return {"saved": False, "done": False, "count": 0, "reason": "Ism yo'q"}
    a = assess(frame)
    db = _f._load(FACE_DB)
    have = db.get(name, [])
    if not a["ok"]:
        return {"saved": False, "done": len(have) >= ENROLL_TARGET,
                "count": len(have), "reason": a["reason"], "px": a.get("px", 0)}
    emb = a["emb"]
    if have:
        top = max(float(np.dot(np.asarray(e, np.float32), emb)) for e in have)
        if top >= ENROLL_NEW_MAX:
            return {"saved": False, "done": len(have) >= ENROLL_TARGET,
                    "count": len(have), "reason": "Boshni biroz buring",
                    "px": a["px"]}
    db.setdefault(name, []).append([float(x) for x in emb])
    _f._save(FACE_DB, db)
    if not _f.has_thumb(name):
        _f._save_thumb(name, a["crop"])
    if branches:
        meta = _f._load(META_DB)
        e = meta.setdefault(name, {})
        e["filiallar"] = sorted(set(e.get("filiallar", [])) | set(branches))
        _f._save(META_DB, meta)
    _cache["mtime"] = None
    n = len(db[name])
    return {"saved": True, "done": n >= ENROLL_TARGET, "count": n,
            "reason": "", "px": a["px"]}


def enroll(frame, name, branches=None):
    name = (name or "").strip()
    if not name:
        return False, "Ism kiritilmadi"
    a = assess(frame)
    if not a["ok"]:
        return False, a["reason"]
    emb = a["emb"]
    db = _f._load(FACE_DB)
    db.setdefault(name, []).append([float(x) for x in emb])
    _f._save(FACE_DB, db)
    if not _f.has_thumb(name):
        _f._save_thumb(name, a["crop"])
    if branches:
        meta = _f._load(META_DB)
        e = meta.setdefault(name, {})
        e["filiallar"] = sorted(set(e.get("filiallar", [])) | set(branches))
        _f._save(META_DB, meta)
    _cache["mtime"] = None
    return True, f"{name}: {len(db[name])}-namuna qo'shildi ({a['px']}px)"


def people():
    db, meta = _f._load(FACE_DB), _f._load(META_DB)
    return sorted(
        ({"name": n, "samples": len(v),
          "branches": meta.get(n, {}).get("filiallar", []),
          "thumb": _f.has_thumb(n)}
         for n, v in db.items()),
        key=lambda p: p["name"])


def remove(name):
    db = _f._load(FACE_DB)
    if name not in db:
        return False, "Bunday xodim yo'q"
    del db[name]
    _f._save(FACE_DB, db)
    try:
        os.remove(_f._thumb_path(name))
    except OSError:
        pass
    meta = _f._load(META_DB)
    if name in meta:
        del meta[name]
        _f._save(META_DB, meta)
    _cache["mtime"] = None
    return True, f"{name} o'chirildi"


def audit():
    """Aralashgan namuna: bir odamning namunalari o'zaro past bo'lsa."""
    db = _f._load(FACE_DB)
    problems = []
    for n, v in db.items():
        if len(v) < 2:
            continue
        embs = [np.asarray(e, np.float32) for e in v]
        lo = min(float(np.dot(embs[i], embs[j]))
                 for i in range(len(embs)) for j in range(i + 1, len(embs)))
        if lo < 0.15:
            problems.append({"type": "mixed", "name": n, "samples": len(v)})
    return problems


# app.py faces._thumb_path ni ham chaqiradi
_thumb_path = _f._thumb_path
