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
# SFace 112x112 kirish bilan ishlaydi. Bundan ancha kichik yuzni
# kattalashtirish yangi ma'lumot qo'shmaydi — ball shovqinga aylanadi.
# Jonli o'lchandi (Yunusobod, 2026-08-24): 13-41 piksellik yuzlar
# 0.21-0.30 ball berdi, ya'ni tanish emas, tasodif. Bunday yuzga ism
# qo'yish — noto'g'ri ism qo'yish demakdir.
#
# Shuning uchun kichik yuzda tanishga URINILMAYDI ham: ism ham berilmaydi,
# qimmat hisob (~138 ms) ham bekorga sarflanmaydi.
MIN_RECOGNIZE_PX = 60

_det = _rec = None
_lock = threading.Lock()


def load_models():
    """Modellarni yuklaydi. ASOSIY IPDA, boshqa modellardan alohida
    chaqirilishi kerak — bir vaqtda yuklash MPS/CoreML ni buzadi."""
    return _models()


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


def _save(path, data):
    """Atomik yozish — yozish yarmida uzilsa baza buzilmasin."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


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
        box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        if box[2] < MIN_RECOGNIZE_PX:
            # Yuz topildi, lekin kim ekanini aytib bo'lmaydi
            out.append({"box": box, "name": None, "score": None,
                        "too_small": True})
            continue
        aligned = rec.alignCrop(frame, f)
        emb = _unit(rec.feature(aligned).flatten())
        best, score = None, -1.0
        for name, embs in known.items():
            s = max(float(np.dot(e, emb)) for e in embs)
            if s > score:
                best, score = name, s
        out.append({
            "box": box,
            "name": best if score >= THRESHOLD else None,
            "score": round(score, 3),
            "too_small": False,
        })
    return out


def annotate(frame, people):
    """identify() natijasini kadrga chizadi (joyida o'zgartiradi)."""
    for p in people:
        x, y, w, h = p["box"]
        named = p["name"] is not None
        color = (0, 200, 0) if named else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if named:
            label = f"{p['name']} {p['score']:.2f}"
        elif p.get("too_small"):
            label = "yuz kichik"          # tanishga urinilmadi
        else:
            label = f"? {p['score']:.2f}"
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


# ── Ro'yxatga olish ──────────────────────────────────────────────────
MIN_ENROLL_PX = 100     # ro'yxatga olishda yuz shundan katta bo'lsin
# Ketma-ket kadrlar deyarli bir xil bo'ladi. Bir xil namunani o'nlab marta
# saqlash bazani shishiradi va tanishga hech narsa qo'shmaydi — foydasi
# TURLI burchakdagi namunalarda. Shundan yuqori o'xshashlik = o'sha kadr.
SAME_SAMPLE = 0.97


def people():
    """Bazadagi xodimlar: [{name, samples, branches}]."""
    db, meta = _load(FACE_DB), _load(META_DB)
    return sorted(
        ({"name": n, "samples": len(v),
          "branches": meta.get(n, {}).get("filiallar", [])}
         for n, v in db.items()),
        key=lambda p: p["name"])


def enroll(frame, name, branches=None):
    """Kadrdagi eng katta yuzni `name` nomiga qo'shadi.

    Bir odamga bir necha marta qo'shish mumkin — turli burchak va yorug'likda
    olingan namunalar tanishni yaxshilaydi.

    Qaytaradi: (ok, xabar)
    """
    name = (name or "").strip()
    if not name:
        return False, "Ism kiritilmadi"

    det, rec = _models()
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, found = det.detect(frame)
    if found is None or len(found) == 0:
        return False, "Kadrda yuz topilmadi"

    # Eng katta yuz — ro'yxatga olayotgan odam kameraga yaqin turadi
    face = max(found, key=lambda f: f[2] * f[3])
    if int(face[2]) < MIN_ENROLL_PX:
        return False, (f"Yuz juda kichik ({int(face[2])}px). "
                       f"Kameraga yaqinroq turing (kamida {MIN_ENROLL_PX}px)")

    emb = _unit(rec.feature(rec.alignCrop(frame, face)).flatten())

    db = _load(FACE_DB)
    # Bu namuna allaqachon bormi (yuz qimirlamagan)
    for e in db.get(name, []):
        if float(np.dot(_unit(e), emb)) >= SAME_SAMPLE:
            return False, "Shu holat allaqachon olingan — yuzni biroz buring"
    # Bu yuz allaqachon boshqa ismga yozilganmi — ogohlantiramiz
    clash, best = None, -1.0
    for other, embs in db.items():
        if other == name:
            continue
        s = max(float(np.dot(_unit(e), emb)) for e in embs)
        if s > best:
            clash, best = other, s

    db.setdefault(name, []).append(emb.tolist())
    _save(FACE_DB, db)

    meta = _load(META_DB)
    if branches:
        entry = meta.setdefault(name, {})
        entry["filiallar"] = sorted(set(entry.get("filiallar", [])) | set(branches))
        _save(META_DB, meta)

    _db_cache["mtime"] = None      # keshni yangilaymiz
    msg = f"{name}: {len(db[name])}-namuna qo'shildi ({int(face[2])}px)"
    if best >= THRESHOLD:
        msg += f" — DIQQAT: {clash} ga ham o'xshaydi ({best:.2f})"
    return True, msg


def remove(name):
    """Xodimni bazadan o'chiradi."""
    db = _load(FACE_DB)
    if name not in db:
        return False, "Bunday xodim yo'q"
    del db[name]
    _save(FACE_DB, db)
    meta = _load(META_DB)
    if name in meta:
        del meta[name]
        _save(META_DB, meta)
    _db_cache["mtime"] = None
    return True, f"{name} o'chirildi"
