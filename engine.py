"""Kamera "ko'zi" — Hermes agent uchun RPC dvigatel (Flask/UI yo'q).

Hermes plagini (~/.hermes/plugins/kamera) shu faylni camera-ai venv'ida
subprocess qilib ochadi va stdin/stdout orqali JSON-qator bilan gaplashadi.
Modellar (YOLO, ArcFace) BIR MARTA yuklanadi va jarayon tirik turguncha
xotirada qoladi. Hermes yopilsa — dvigatel ham yopiladi.

So'rov:  {"id": 1, "cmd": "kim_bor", "kamera": "B4"}
Javob:   {"id": 1, "ok": true, ...}

Buyruqlar:
  kameralar                       — mavjud kameralar ro'yxati
  kadr      kamera                — bitta kadr, JPEG fayl yo'li
  kim_bor   kamera                — odam soni + tanilgan xodimlar + begonalar
  kuzat     kamera, soniya        — hodisa kelguncha (yoki vaqt tugaguncha)
                                    kuzatadi: xodim keldi / begona keldi /
                                    begona qaytdi. Birinchi hodisada qaytadi.
  enroll    kamera, ism, rol?     — kameradagi eng katta yuzni bazaga yozadi

Qo'lda sinov:
  echo '{"id":1,"cmd":"kim_bor","kamera":"B4"}' | BRANCHES=Yunusobod ./venv/bin/python engine.py
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# Modellar (ultralytics/insightface) stdout'ga yozadi — JSON kanalini buzmasin.
_OUT = sys.stdout
sys.stdout = sys.stderr
os.environ.setdefault("MAC_CAMERA_OFF", "1")
os.environ.setdefault("TESTCAM", "0")

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402
import requests                              # noqa: E402
from requests.auth import HTTPDigestAuth     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import nvr                                   # noqa: E402
import pose                                  # noqa: E402
import arcface as faces                      # noqa: E402
import visitors                              # noqa: E402
import tracker                               # noqa: E402
import identity                              # noqa: E402
import body                                  # noqa: E402

OUT_DIR = os.path.join(_HERE, "data", "engine")
os.makedirs(OUT_DIR, exist_ok=True)
STAFF_GAP = 10 * 60          # xodim shuncha vaqt ko'rinmasa — qayta "keldi"
POLL = 1.5                   # NVR ~1 yangi kadr/sek; tezroq so'rash foydasiz
STRANGER_HITS = visitors.STRANGER_MIN_HITS

_log = lambda *a: print("[engine]", *a, file=sys.stderr, flush=True)  # noqa: E731

# ── kameralar ────────────────────────────────────────────────────────
_branches = {}
_sess = {}


def _branch(name):
    if name not in _branches:
        _branches[name] = nvr.Branch(name, nvr.BRANCH_HOSTS[name],
                                     nvr.BRANCH_CAMERAS[name])
        s = requests.Session()
        s.auth = HTTPDigestAuth(nvr.USER, _branches[name].password)
        _sess[name] = s
    return _branches[name]


def cameras():
    out = []
    for br, cams in nvr.BRANCH_CAMERAS.items():
        if br not in nvr.ENABLED:
            continue
        for ch, nm in cams.items():
            out.append({"kamera": nm, "filial": br, "kanal": ch})
    return out


def _find(name):
    """'B4' yoki 'Yunusobod/1101' yoki '1101' → Camera."""
    name = (name or "").strip()
    for c in cameras():
        if name.lower() in (c["kamera"].lower(), c["kanal"],
                            f"{c['filial']}/{c['kanal']}".lower()):
            br = _branch(c["filial"])
            return br.cameras[c["kanal"]]
    raise ValueError(f"Kamera topilmadi: {name}. Mavjud: "
                     + ", ".join(c["kamera"] for c in cameras()))


def _grab(cam):
    """Bitta YANGI kadr (JPEG bayt). NVR o'sha rasmni qaytarsa — kutadi."""
    br = cam.branch
    for _ in range(6):
        data = br.get(_sess[br.name], cam.url)
        if data is not None and cam.fetch_once_bytes(data):
            return data
        time.sleep(0.5)
    return None


def _patch_camera():
    """nvr.Camera.fetch_once sessiya ichida so'raydi; bizga baytdan
    'yangi/o'sha' ajratish yetadi."""
    def fetch_once_bytes(self, data):
        import hashlib
        d = hashlib.md5(data).digest()
        if d == self._last_digest:
            return False
        self._last_digest = d
        return True
    nvr.Camera.fetch_once_bytes = fetch_once_bytes


_patch_camera()


def _save(jpeg, tag):
    path = os.path.join(OUT_DIR, f"{tag}_{time.strftime('%H%M%S')}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg)
    return path


def _draw(frame, persons, found, strangers):
    for p in persons:
        x1, y1, x2, y2 = [int(t) for t in p["box"]]
        if p.get("name"):
            col, label = (0, 200, 0), p["name"]
        elif p.get("visitor"):
            col, label = (0, 140, 255), f"Begona #{p['visitor']}"
        else:
            col, label = (255, 160, 0), ""
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        if label:
            cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for f in found:
        if f.get("how", "yuz") != "yuz":
            continue
        x, y, w, h = f["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else b""


# ── tahlil ───────────────────────────────────────────────────────────
_staff_seen = {}     # {ism: oxirgi ko'rilgan vaqt}
_ident = {}          # {kamera: identity.CamIdentity} — app.py bilan BIR XIL mantiq


def analyze(cam, jpeg, track=True):
    """Bitta kadr: odamlar, xodimlar, begonalar (yuz + tana, identity.py qoidalari).

    ByteTrack trek beradi, CamIdentity trekni xodim/mehmonga bog'laydi.
    Mehmon yozuvi (raqam, vektorlar) app.py bilan umumiy data/visitors.json da.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    persons = pose.people_in(frame)
    persons = tracker.get(cam.key).update(persons)
    visitors.STRANGER_CAMS.add(cam.name)          # dvigatelda har kamera nazoratda
    ident = _ident.get(cam.name)
    if ident is None:
        ident = _ident[cam.name] = identity.CamIdentity(cam.name)
    res = faces.identify(frame, branch=cam.branch.name)
    small = sum(1 for f in res if f.get("too_small"))
    found, strangers, new_visitors, reseen = ident.step(frame, persons, res, True)
    now = time.time()
    events = []
    for f in found:
        last = _staff_seen.get(f["name"], 0)
        _staff_seen[f["name"]] = now
        if track and now - last > STAFF_GAP:
            events.append({"tur": "xodim_keldi", "kim": f["name"], "bal": f["score"],
                           "qanday": f.get("how", "yuz")})
    for v in new_visitors:
        events.append({"tur": "begona_keldi", "n": v["n"],
                       "yuz": bool(v.get("embs")), "tana": bool(v.get("body_embs"))})
    for v in reseen:
        events.append({"tur": "begona_qaytdi", "n": v["n"],
                       "birinchi": time.strftime("%H:%M", time.localtime(v["first"]))})
    seen_n = {}
    for s in strangers:
        n = int(s["name"].split("#")[1])
        seen_n[n] = {"n": n, "bal": s["score"], "qanday": s.get("how"),
                     "yangi": any(v["n"] == n for v in new_visitors)}
    return {"kamera": cam.name, "odam": len(persons),
            "xodimlar": [{"ism": f["name"], "bal": f["score"], "qanday": f.get("how", "yuz")}
                         for f in found],
            "begonalar": list(seen_n.values()),
            "kichik_yuz": small, "hodisalar": events,
            "_frame": frame, "_persons": persons, "_found": found,
            "_strangers": strangers}


def _public(a, jpeg_path=None):
    out = {k: v for k, v in a.items() if not k.startswith("_")}
    if jpeg_path:
        out["rasm"] = jpeg_path
    return out


# ── buyruqlar ────────────────────────────────────────────────────────
def cmd_kameralar(req):
    return {"kameralar": cameras()}


def cmd_kadr(req):
    cam = _find(req.get("kamera"))
    jpeg = _grab(cam)
    if jpeg is None:
        return {"ok": False, "xato": f"{cam.name}: kadr kelmadi (NVR)"}
    return {"kamera": cam.name, "rasm": _save(jpeg, cam.name)}


def cmd_kim_bor(req):
    cam = _find(req.get("kamera"))
    jpeg = _grab(cam)
    if jpeg is None:
        return {"ok": False, "xato": f"{cam.name}: kadr kelmadi (NVR)"}
    a = analyze(cam, jpeg, track=False)
    vis = _draw(a["_frame"], a["_persons"], a["_found"], a["_strangers"])
    return _public(a, _save(vis, cam.name + "_kimbor"))


def cmd_kuzat(req):
    cam = _find(req.get("kamera"))
    secs = min(float(req.get("soniya", 300)), 900)
    t0 = time.time()
    last = None
    n = 0
    while time.time() - t0 < secs:
        jpeg = _grab(cam)
        if jpeg is None:
            time.sleep(POLL)
            continue
        n += 1
        a = analyze(cam, jpeg, track=True)
        last = a
        if a["hodisalar"]:
            vis = _draw(a["_frame"], a["_persons"], a["_found"], a["_strangers"])
            out = _public(a, _save(vis, cam.name + "_hodisa"))
            out.update(kadrlar=n, davomiylik=round(time.time() - t0))
            return out
        time.sleep(POLL)
    out = _public(last) if last else {"kamera": cam.name, "hodisalar": []}
    out.update(kadrlar=n, davomiylik=round(time.time() - t0),
               izoh="Vaqt tugadi, hodisa bo'lmadi")
    return out


def cmd_enroll(req):
    cam = _find(req.get("kamera"))
    ism = (req.get("ism") or "").strip()
    if not ism:
        return {"ok": False, "xato": "ism kerak"}
    jpeg = _grab(cam)
    if jpeg is None:
        return {"ok": False, "xato": f"{cam.name}: kadr kelmadi"}
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    r = faces.enroll_sample(frame, ism, branches=[cam.branch.name])
    if r.get("saved") and req.get("rol"):
        import faces as _f
        meta = _f._load(_f.META_DB)
        meta.setdefault(ism, {})["role"] = req["rol"]
        _f._save(_f.META_DB, meta)
    return r


COMMANDS = {"kameralar": cmd_kameralar, "kadr": cmd_kadr, "kim_bor": cmd_kim_bor,
            "kuzat": cmd_kuzat, "enroll": cmd_enroll}


def main():
    _log("modellar yuklanmoqda…")
    pose.people_in(np.zeros((360, 640, 3), np.uint8))    # YOLO isitish
    faces.load_models()
    body.load_models()
    _log("tayyor")
    _OUT.write(json.dumps({"ready": True}) + "\n"); _OUT.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            fn = COMMANDS.get(req.get("cmd"))
            if fn is None:
                res = {"ok": False, "xato": f"noma'lum buyruq: {req.get('cmd')}"}
            else:
                res = fn(req)
                res.setdefault("ok", True)
        except Exception as e:                       # noqa: BLE001
            res = {"ok": False, "xato": f"{type(e).__name__}: {e}"}
        res["id"] = req.get("id") if isinstance(req, dict) else None
        _OUT.write(json.dumps(res, ensure_ascii=False) + "\n"); _OUT.flush()


if __name__ == "__main__":
    main()
