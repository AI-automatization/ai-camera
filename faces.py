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
THUMB_DIR = os.path.join(DATA_DIR, "faces_thumbs")   # avatar rasmlar (ro'yxat uchun)
os.makedirs(THUMB_DIR, exist_ok=True)
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
# 50 ga tushirildi: jonli o'lchandi (Yunusobod Coworking 1), yuz 55px da
# Sardor 0.57 ball berdi — ishonchli. 40-44px da esa ball 0.2 (shovqin),
# lekin THRESHOLD (0.40) ularni baribir rad etadi. Ya'ni 50px chegara
# haqiqiy tanishlarni o'tkazadi, yolg'onni THRESHOLD to'sadi.
MIN_RECOGNIZE_PX = 45

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
# ── Face ID kabi qo'shish: sifat va xilma-xillik nazorati ────────────
ENROLL_SHARP_MIN = 15      # yuz tiniqligi (Laplacian var) — faqat ANIQ xira
                           # kadrni rad etadi; asosiy sifatni o'lcham+xilma-
                           # xillik ta'minlaydi (juda qattiq bo'lsa hech tugamaydi)
ENROLL_NEW_MAX = 0.92      # yangi namuna mavjuddan shu qadar farq qilsin
                           # (0.92 dan yuqori = o'sha burchak, qo'shmaymiz)
ENROLL_SAME_MIN = 0.30     # lekin butunlay boshqa odam bo'lmasin
ENROLL_TARGET = 14         # shuncha xilma-xil sifatli namuna yetarli


def people():
    """Bazadagi xodimlar: [{name, samples, branches}]."""
    db, meta = _load(FACE_DB), _load(META_DB)
    return sorted(
        ({"name": n, "samples": len(v),
          "branches": meta.get(n, {}).get("filiallar", []),
          "thumb": has_thumb(n)}
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
    fx, fy, fw2, fh2 = (int(v) for v in face[:4])
    _crop = frame[max(0, fy):fy + fh2, max(0, fx):fx + fw2]

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
    if not has_thumb(name) and _crop.size:
        _save_thumb(name, _crop)

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
    try:
        os.remove(_thumb_path(name))
    except OSError:
        pass
    meta = _load(META_DB)
    if name in meta:
        del meta[name]
        _save(META_DB, meta)
    _db_cache["mtime"] = None
    return True, f"{name} o'chirildi"


def audit():
    """Bazadagi muammolarni topadi.

    Yuz tanish sifati bazaga bog'liq. Ikki xil nosozlik jimgina hamma
    narsani buzadi:

      * BIR ODAM IKKI ISMDA — ikkalasi bir-biriga chegaradan yuqori
        o'xshaydi, ya'ni kamera qaysi birini aytishini oldindan bilib
        bo'lmaydi.
      * ARALASHGAN NAMUNA — bitta ismda turli odamlarning yuzi.
        Namunalarning o'zaro o'xshashligi juda past bo'lib qoladi.

    Qaytaradi: [{type, ...}] ro'yxati.
    """
    db = _load(FACE_DB)
    known = {n: [_unit(e) for e in v] for n, v in db.items()}
    problems = []

    names = sorted(known)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            best = max(float(np.dot(x, y)) for x in known[a] for y in known[b])
            if best >= THRESHOLD:
                problems.append({"type": "duplicate", "a": a, "b": b,
                                 "score": round(best, 2)})

    for n, embs in known.items():
        if len(embs) < 3:
            continue
        sims = [float(np.dot(embs[i], embs[j]))
                for i in range(len(embs)) for j in range(i + 1, len(embs))]
        worst = min(sims)
        if worst < 0.10:
            problems.append({"type": "mixed", "name": n,
                             "samples": len(embs), "worst": round(worst, 2)})
    return problems


# ── Avatar rasm (ro'yxatda ko'rsatish uchun) ─────────────────────────
def _thumb_path(name):
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    return os.path.join(THUMB_DIR, f"{safe}.jpg")


def has_thumb(name):
    return os.path.exists(_thumb_path(name))


def _save_thumb(name, crop):
    """Yuz kirqimini ~200px avatar qilib saqlaydi (mavjud bo'lsa yozmaydi)."""
    try:
        h, w = crop.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        sq = crop[y0:y0 + side, x0:x0 + side]
        sq = cv2.resize(sq, (200, 200), interpolation=cv2.INTER_AREA)
        cv2.imwrite(_thumb_path(name), sq, [cv2.IMWRITE_JPEG_QUALITY, 88])
    except Exception:
        pass


# ── Face ID uslubidagi ro'yxatga olish ───────────────────────────────
def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(frame):
    """Kadrdagi eng katta yuzni baholaydi (saqlamaydi).

    Qaytaradi: {ok, reason, px, sharp, emb}. ok=False bo'lsa reason nima
    yetishmayotganini aytadi — brauzer foydalanuvchiga ko'rsatadi.
    """
    det, rec = _models()
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, found = det.detect(frame)
    if found is None or len(found) == 0:
        return {"ok": False, "reason": "Yuz ko'rinmayapti", "px": 0}
    face = max(found, key=lambda f: f[2] * f[3])
    px = int(face[2])
    if px < MIN_ENROLL_PX:
        return {"ok": False, "reason": f"Yaqinroq keling ({px}px)", "px": px}
    x, y, fw, fh = (int(v) for v in face[:4])
    crop = frame[max(0, y):y + fh, max(0, x):x + fw]
    if crop.size == 0:
        return {"ok": False, "reason": "Yuz kadr chetida", "px": px}
    sharp = _sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    if sharp < ENROLL_SHARP_MIN:
        return {"ok": False, "reason": "Xira — sekinroq harakatlaning",
                "px": px, "sharp": round(sharp)}
    emb = _unit(rec.feature(rec.alignCrop(frame, face)).flatten())
    return {"ok": True, "reason": "", "px": px, "sharp": round(sharp),
            "emb": emb, "crop": crop}


def enroll_sample(frame, name, branches=None):
    """Bitta kadrni sifat va XILMA-XILLIK bo'yicha tekshirib saqlaydi.

    Face ID kabi: sifatsiz yoki allaqachon olingan burchakni RAD etadi,
    faqat yangi, sifatli namunani qo'shadi.

    Qaytaradi: {saved, done, count, reason} — count: shu odamda nechta
    namuna bor, done: yetarli (ENROLL_TARGET) bo'ldimi.
    """
    name = (name or "").strip()
    if not name:
        return {"saved": False, "done": False, "count": 0, "reason": "Ism yo'q"}
    a = assess(frame)
    db = _load(FACE_DB)
    have = db.get(name, [])
    if not a["ok"]:
        return {"saved": False, "done": len(have) >= ENROLL_TARGET,
                "count": len(have), "reason": a["reason"], "px": a.get("px", 0)}
    emb = a["emb"]
    # Mavjud namunalarga o'xshashlik: juda o'xshasa — o'sha burchak, qo'shmaymiz
    if have:
        top = max(float(np.dot(_unit(e), emb)) for e in have)
        if top >= ENROLL_NEW_MAX:
            return {"saved": False, "done": len(have) >= ENROLL_TARGET,
                    "count": len(have), "reason": "Boshni biroz buring",
                    "px": a["px"]}
    db.setdefault(name, []).append(emb.tolist())
    _save(FACE_DB, db)
    if not has_thumb(name):          # birinchi (odatda frontal) namuna = avatar
        _save_thumb(name, a["crop"])
    if branches:
        meta = _load(META_DB)
        e = meta.setdefault(name, {})
        e["filiallar"] = sorted(set(e.get("filiallar", [])) | set(branches))
        _save(META_DB, meta)
    _db_cache["mtime"] = None
    n = len(db[name])
    return {"saved": True, "done": n >= ENROLL_TARGET, "count": n,
            "reason": "", "px": a["px"]}
