"""Mehmon xotirasi — begona (bazada yo'q) odamni vaqtincha eslab qolish.

Muammo: mijoz kameraga kirdi → "begona" → Telegram. 5 daqiqa yo'qoldi,
qaytib keldi → ByteTrack yangi trek beradi → yana "begona" → yana Telegram.
Bitta odam uchun bir necha xabar — reseptsiya uchun shovqin.

Yechim: begona yuzning ArcFace vektorini ismsiz, muddatli (TTL) ro'yxatga
yozamiz. Keyingi begona yuz avval shu ro'yxat bilan solishtiriladi —
mos kelsa, o'sha mehmon, xabar yo'q ("yana ko'rindi" deb ichki yoziladi).
Ertaga (TTL o'tgach) yana kelsa — yangi tashrif, yangi xabar.

Baza: data/visitors.json (restartdan omon qoladi). Keyin DB'ga o'tadi.

Sozlash (env):
  STRANGER_CAMS="B4"          qaysi kameralarda begona nazorati (bo'sh = o'chiq)
  VISITOR_THRESHOLD=0.28      mehmon o'zi bilan mosligi (xodimnikidan pastroq —
                              xato "o'sha odam" deyish xato "yangi begona"dan arzon)
  VISITOR_TTL_H=12            mehmon yozuvi necha soat yashaydi
  STRANGER_MAX_STAFF=0.22     xodim bali shundan PAST bo'lsagina begona nomzodi
                              (0.22–0.30 oralig'i — noaniq, hech narsa demaymiz)
  STRANGER_MIN_HITS=3         necha marta (~1s oralig'ida) ko'rilsin, keyin qaror
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (.env dan ham o'qiladi)
"""
import json
import os
import threading
import time
import uuid

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
VISITORS_DB = os.path.join(DATA_DIR, "visitors.json")
THUMB_DIR = os.path.join(DATA_DIR, "visitors_thumbs")   # panel uchun yuz kesmalari
HISTORY_DAYS = 30            # panel tarixi shuncha kun saqlanadi (TTL o'tsa ham)


def _load_env(path=os.path.join(_HERE, ".env")):
    """python-dotenv o'rnatilmagan — oddiy KEY=VALUE o'qish. env ustun."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_env()

STRANGER_CAMS = {c.strip() for c in os.environ.get("STRANGER_CAMS", "").split(",")
                 if c.strip()}
THRESHOLD = float(os.environ.get("VISITOR_THRESHOLD", "0.28"))
TTL_SEC = float(os.environ.get("VISITOR_TTL_H", "12")) * 3600
STRANGER_MAX_STAFF = float(os.environ.get("STRANGER_MAX_STAFF", "0.27"))   # 0.27-0.30 noaniq (tor)
STRANGER_MIN_HITS = int(os.environ.get("STRANGER_MIN_HITS", "3"))
MAX_EMBS = 8            # bir mehmondan ko'pi bilan shuncha vektor (turli burchak, 5-8s turganda)
NEW_EMB_MAX = 0.90      # yangi vektor mavjuddan shu qadar farq qilsa saqlanadi
RESEEN_GAP = 60.0       # shundan uzoq uzilishdan keyin qaytsa — "yana ko'rindi"
# Yuz SIFATI darvozasi — sifatsiz yuz (qorong'i siluet, boshi pastda, xira)
# "umumiy" vektor beradi: boshqa odamlar bir-biriga 0.3-0.5 o'xshab, bir odam
# 7 ta mehmon bo'lib ketadi (2026-09-09 Mac sinovida o'lchandi). Shuning uchun
# mehmon faqat TOZA yuzdan yoziladi.
MIN_PX = int(os.environ.get("STRANGER_MIN_PX", "60"))
SHARP_MIN = 3                        # webkamera JPEG kadri: Sardor jonli 8, bola 3 o'lchandi
BRIGHT_MIN, BRIGHT_MAX = 45, 215
YAW_MAX, PITCH_MAX = 35, 30          # bosh burilishi (daraja)
HIT_SAME_MIN = 0.45                  # 3 ta hit bir odam bo'lsin (o'zaro o'xshashlik)
# Tana (ReID) — body.py da o'lchangan: bir odam 0.74, boshqa odam ≤0.50 (CLIP)
BODY_THRESHOLD = float(os.environ.get("BODY_THRESHOLD", "0.65"))
BODY_HIT_SAME_MIN = 0.60             # hitlar orasida tana mosligi
BODY_AMBIG_MIN = float(os.environ.get("BODY_AMBIG_MIN", "0.45"))   # 0.45–0.65: noaniq → yangi raqam BERILMAYDI, kutiladi (orqa ko'rinish ham shu yerga tushadi)
BODY_ONLY_MIN_H = int(os.environ.get("BODY_ONLY_MIN_H", "160"))    # yuzsiz raqam uchun quti balandligi
BODY_ONLY_MIN_ASPECT = 1.3           # yuzsiz raqam faqat TIK turgan odamga (h/w); o'tirgan — yuz kutiladi
# Yuzsiz YANGI raqam berish (0 = yo'q). Sardor 2026-09-11: oldi bilan kelib, orqasi bilan
# qaytgan odam yangi mijoz bo'lib ketdi (orqa ko'rinish vektori boshqacha). Shuning uchun
# tana FAQAT qayta tanish uchun; birinchi yozuv yuz bilan. Orqasi bilan turgan — kutiladi.
BODY_ONLY_REGISTER = os.environ.get("BODY_ONLY_REGISTER", "1") == "1"   # Sardor: tana bilan ishlasin
CROWD_OVERLAP = 0.25                 # quti boshqa quti bilan shunchadan ko'p kesishsa — "olomon"
CROWD_BODY_THRESHOLD = 0.75          # olomonda tana mosligi QATTIQROQ (qo'shni odam kesmaga kiradi)
BODY_UNBIND_MAX = 0.40               # bog'langan trekning tanasi shundan past bo'lsa — mos emas
UNBIND_STRIKES = 3                   # ketma-ket shuncha marta mos kelmasa — trek uziladi (ID almashgan)
FACE_CONTRA = 0.15                   # yuzlar shundan past — aniq boshqa odam (tana o'xshasa ham)
MAX_BODY_EMBS = 8
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")


def enabled(camera_name):
    return camera_name in STRANGER_CAMS


def good_face(f):
    """Mehmon vektori uchun yuz yaroqlimi. (ok, sabab)."""
    if f["box"][2] < MIN_PX:
        return False, "kichik"
    if f.get("sharp", 0) < SHARP_MIN:
        return False, "xira"
    b = f.get("bright", 0)
    if b < BRIGHT_MIN or b > BRIGHT_MAX:
        return False, "qorong'i" if b < BRIGHT_MIN else "yorug'"
    p = f.get("pose")
    if p and (abs(p[1]) > YAW_MAX or abs(p[0]) > PITCH_MAX):
        return False, "burilgan"
    return True, ""


class Memory:
    """Mehmonlar ro'yxati. Har yozuv:
    {id, embs[[512]...], first, last, camera, seen:[ts...], tg_msg_id, n}
    """

    def __init__(self, path=VISITORS_DB):
        self.path = path
        self._lock = threading.Lock()
        self._items = []
        self._load()

    # ── saqlash ──────────────────────────────────────────────────────
    def _load(self):
        try:
            with open(self.path) as f:
                self._items = json.load(f)
        except (OSError, ValueError):
            self._items = []
        self._prune()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._items, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _prune(self, now=None):
        """Faylda 30 kunlik TARIX qoladi; TTL faqat solishtirishda (active)."""
        now = now or time.time()
        keep = HISTORY_DAYS * 86400
        gone = [v for v in self._items if now - v["last"] >= keep]
        for v in gone:
            try:
                os.remove(thumb_path(v["id"]))
            except OSError:
                pass
        self._items = [v for v in self._items if now - v["last"] < keep]

    def _active_items(self, now):
        return [v for v in self._items if now - v["last"] < TTL_SEC]

    # ── so'rovlar ────────────────────────────────────────────────────
    def active(self, now=None):
        now = now or time.time()
        with self._lock:
            self._prune(now)
            return [dict(v, embs=None) for v in self._active_items(now)]

    def history(self, date=None):
        """Panel uchun: shu kunda ko'rilgan mehmonlar (yangi birinchi), vektorsiz."""
        with self._lock:
            out = []
            for v in self._items:
                days = {time.strftime("%Y-%m-%d", time.localtime(t)) for t in v["seen"]}
                if date and date not in days:
                    continue
                out.append({"id": v["id"], "n": v["n"], "camera": v["camera"],
                            "day": v.get("day") or time.strftime("%Y-%m-%d", time.localtime(v["first"])),
                            "first": v["first"], "last": v["last"],
                            "returns": max(0, len(v["seen"]) - 1),
                            "telegram": bool(v.get("tg_msg_id")),
                            "active": time.time() - v["last"] < TTL_SEC,
                            "thumb": os.path.exists(thumb_path(v["id"]))})
            out.sort(key=lambda x: x["last"], reverse=True)
            return out

    def dates(self):
        with self._lock:
            ds = {time.strftime("%Y-%m-%d", time.localtime(t))
                  for v in self._items for t in v["seen"]}
        return sorted(ds, reverse=True)

    def get(self, vid):
        with self._lock:
            return next((v for v in self._items if v["id"] == vid), None)

    @staticmethod
    def _best(embs, q):
        if q is None or not embs:
            return None
        q = np.asarray(q, np.float32)
        return max(float(np.dot(np.asarray(e, np.float32), q)) for e in embs)

    def match(self, face_emb=None, body_emb=None, geom=None, exclude=None, now=None, body_thr=None):
        """Eng mos FAOL mehmon. Qaytaradi (v | None, qanday, bal).

        Tartib: yuz mos → o'sha odam. Yuz aniq farq qilsa (FACE_CONTRA) → tana
        o'xshasa ham BOSHQA odam. Yuz yo'q/noaniq → tana (yuqori chegara) +
        qomat qarama-qarshi bo'lmasin. exclude — hozir ekranda boshqa trekka
        bog'langan mehmonlar (bir vaqtda ikki odam = ikki odam).
        """
        import body as _body
        now = now or time.time()
        exclude = exclude or set()
        body_thr = body_thr if body_thr is not None else BODY_THRESHOLD
        with self._lock:
            self._prune(now)
            best, how, score = None, "", -1.0
            for v in self._active_items(now):
                if v["id"] in exclude:
                    continue
                fs = self._best(v.get("embs"), face_emb)
                bs = self._best(v.get("body_embs"), body_emb)
                if fs is not None and fs >= THRESHOLD:
                    if how != "yuz" or fs > score:
                        best, how, score = v, "yuz", fs
                    continue
                if how == "yuz":
                    continue                       # yuz mosi bor — tana nomzodlar kerak emas
                if fs is not None and fs < FACE_CONTRA:
                    continue                       # yuzlar aniq farq — boshqa odam
                if bs is not None and bs >= body_thr:
                    if _body.geometry_conflict(v.get("geom"), geom):
                        continue                   # qomat qarama-qarshi
                    if bs > score:
                        best, how, score = v, "tana", bs
        if best is None:
            return None, "", round(max(score, 0.0), 3)
        return best, how, round(score, 3)

    def body_best(self, body_emb, exclude=None, now=None):
        """Faol mehmonlar ichida eng yuqori tana o'xshashligi (chegarasiz)."""
        if body_emb is None:
            return 0.0
        now = now or time.time()
        exclude = exclude or set()
        with self._lock:
            best = 0.0
            for v in self._active_items(now):
                if v["id"] in exclude:
                    continue
                s = self._best(v.get("body_embs"), body_emb)
                if s is not None and s > best:
                    best = s
        return round(best, 3)

    def add(self, emb, camera, now=None, body_emb=None, geom=None):
        """Yangi mehmon (yuz va/yoki tana vektori bilan)."""
        now = now or time.time()
        v = {"id": uuid.uuid4().hex[:8],
             "embs": [[float(x) for x in emb]] if emb is not None else [],
             "body_embs": [[float(x) for x in body_emb]] if body_emb is not None else [],
             "geom": geom,
             "first": now, "last": now, "camera": camera, "seen": [now],
             "tg_msg_id": None, "n": 1}
        with self._lock:
            # KUNLIK raqam: har kun 1 dan ("bugun 12-mijoz"). Kun = birinchi ko'rilgan sana
            day = time.strftime("%Y-%m-%d", time.localtime(now))
            v["day"] = day
            v["n"] = max((x["n"] for x in self._items
                          if x.get("day", time.strftime("%Y-%m-%d", time.localtime(x["first"]))) == day),
                         default=0) + 1
            self._items.append(v)
            self._save()
        return v

    def seen(self, v, emb=None, camera=None, now=None, body_emb=None, geom=None):
        """Mehmon yana kadrda. Qaytaradi: bu uzilishdan keyingi QAYTISHMI."""
        now = now or time.time()
        with self._lock:
            reseen = now - v["last"] > RESEEN_GAP
            if reseen:
                v["seen"].append(now)
                v["seen"] = v["seen"][-20:]
            v["last"] = now
            if camera:
                v["camera"] = camera
            if geom and not v.get("geom"):
                v["geom"] = geom
            if emb is not None and len(v["embs"]) < MAX_EMBS:
                e = np.asarray(emb, np.float32)
                top = self._best(v["embs"], e)
                if top is None or top < NEW_EMB_MAX:
                    v["embs"].append([float(x) for x in e])
            if body_emb is not None and len(v.setdefault("body_embs", [])) < MAX_BODY_EMBS:
                e = np.asarray(body_emb, np.float32)
                top = self._best(v["body_embs"], e)
                if top is None or top < 0.95:
                    v["body_embs"].append([float(x) for x in e])
            if reseen or len(v["seen"]) == 1:
                self._save()
        return reseen

    def set_tg(self, v, msg_id):
        with self._lock:
            v["tg_msg_id"] = msg_id
            self._save()


MEMORY = Memory()


def thumb_path(vid):
    return os.path.join(THUMB_DIR, f"{vid}.jpg")


def save_thumb(v, frame, box):
    """Mehmon kesmasi (panel uchun). box = (x1, y1, x2, y2) — butun odam."""
    import cv2
    x1, y1, x2, y2 = [int(t) for t in box]
    w, h = x2 - x1, y2 - y1
    m = int(w * 0.10)
    H, W = frame.shape[:2]
    crop = frame[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)]
    if crop.size == 0:
        return False
    crop = cv2.resize(crop, (160, int(160 * crop.shape[0] / crop.shape[1])),
                      interpolation=cv2.INTER_AREA)
    os.makedirs(THUMB_DIR, exist_ok=True)
    return bool(cv2.imwrite(thumb_path(v["id"]), crop, [cv2.IMWRITE_JPEG_QUALITY, 88]))


# ── Telegram ─────────────────────────────────────────────────────────
def _hm(ts):
    return time.strftime("%H:%M", time.localtime(ts))


def caption(v):
    lines = [f"Begona mijoz #{v['n']} — {v['camera']}",
             f"Keldi: {time.strftime('%d.%m', time.localtime(v['first']))} {_hm(v['first'])}"]
    again = v["seen"][1:]
    if again:
        lines.append("Yana ko'rindi: " + ", ".join(_hm(t) for t in again[-5:]))
    return "\n".join(lines)


def _tg(method, **kw):
    import requests
    files = kw.pop("files", None)
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                      data=kw, files=files, timeout=15)
    j = r.json()
    if not j.get("ok"):
        print(f"[visitors] telegram {method}: {j.get('description')}")
    return j.get("result")


def notify_new(v, jpeg):
    """Yangi mehmon — rasm bilan bitta xabar. Analizatorni bloklamaydi."""
    if not (TG_TOKEN and TG_CHAT):
        print(f"[visitors] telegram sozlanmagan — #{v['n']} xabar yuborilmadi")
        return

    def go():
        try:
            res = _tg("sendPhoto", chat_id=TG_CHAT, caption=caption(v),
                      files={"photo": ("begona.jpg", jpeg, "image/jpeg")})
            if res:
                MEMORY.set_tg(v, res["message_id"])
        except Exception as e:
            print(f"[visitors] telegram xato: {e}")
    threading.Thread(target=go, daemon=True).start()


def notify_reseen(v):
    """Qaytib keldi — YANGI xabar emas, eskisining matni yangilanadi."""
    if not (TG_TOKEN and TG_CHAT and v.get("tg_msg_id")):
        return

    def go():
        try:
            _tg("editMessageCaption", chat_id=TG_CHAT,
                message_id=v["tg_msg_id"], caption=caption(v))
        except Exception as e:
            print(f"[visitors] telegram edit xato: {e}")
    threading.Thread(target=go, daemon=True).start()
