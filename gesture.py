"""Qo'l ko'tarish signalini aniqlash + Telegram xabar.

Nega barmoq emas: B4 kabi shift kameralarida qo'l kadrda 10-15 piksel bo'ladi —
MediaPipe barmoqlarni umuman ko'rmaydi (6x kattalashtirib sinaldi, topmadi).
Pose (yelka/bilak/bosh) nuqtalari esa 0.86-0.96 ishonch bilan topiladi,
shuning uchun signal = "bilak boshdan yuqorida, bir necha sekund".

app.py shuni ishlatadi. Mustaqil sinash:
    ./venv/bin/python gesture.py [kanal]
"""
import os
import threading
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---- Sozlamalar ----
POSE_MODEL = "yolov8s-pose.pt"   # odam + 17 ta tana nuqtasi (bitta o'tishda)
POSE_IMGSZ = 960
POSE_CONF = 0.25
KP_CONF = 0.30          # nuqta ishonchi shundan past bo'lsa hisobga olinmaydi
# Chegaralar jonli o'lchovdan olingan (2026-08-03), hammasi odam bo'yiga nisbatan.
#   B4, qo'l ko'tarilgan:  bilak-yelka +0.02..+0.48, bilak-burun -0.10..+0.43
#   B4, qo'l tushirilgan:  bilak-yelka -0.10..-0.27
#   B2, YOLG'ON signallar: bilak-yelka +0.07..+0.22, bilak-burun -0.15..-0.23,
#                          tirsak-yelka -0.07..+0.01
# Faqat bilak-yelka yetarli emas — B2 da stolga engashgan odam ham shu oraliqqa tushadi.
# Shuning uchun qo'shimcha shart: qo'l bosh balandligida BO'LSIN yoki tirsak ko'tarilsin.
RAISE_MARGIN = 0.10     # bilak yelkadan shuncha yuqori (1-shart)
# 2-shart: bilak odamning ENG TEPA qismi bo'lsin. Qo'l ko'tarilganda bilak
# boshdan ham yuqori chiqadi va odam qutisining tepasiga tegadi.
# O'lchandi (A5, iyagiga qo'l tirab o'tirganlar — YOLG'ON signal berardi):
#   bilak quti tepasidan 0.33-0.39 past. Haqiqiy ko'tarishda ~0.
# Burun/tirsak sharti yetarli emas edi: iyakka tiralgan qo'l ham ularni
# qanoatlantirardi.
TOP_FRAC = 0.15         # bilak quti tepasining shuncha ulushi ichida bo'lsin
# Imo tekshiruvi odam filtridan zaif bo'lmasin: aks holda stuldagi ryukzakni
# "odam" deb o'qib, unda "qo'l" topadi (B3 da shunday yolg'on signal chiqdi).
SHOULDER_MIN = 0.80     # yelka nuqtasi ishonchi
STRONG_KP_MIN = 6       # kamida shuncha ishonchli tana nuqtasi
PERSON_MIN_H = 80       # bundan kichik odamda imo o'qish ishonchsiz
HOLD_SEC = 2.0          # qo'l shuncha vaqt ko'tarilib turishi kerak (tasodifga qarshi)
GAP_TOL = 1.0           # shu vaqtgacha uzilish "qo'l tushdi" deb hisoblanmaydi
COOLDOWN_SEC = 60       # bitta signaldan keyin shuncha vaqt jim turadi

# pose nuqta indekslari (COCO)
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10

TELEGRAM_ENV = os.path.expanduser("~/.claude/channels/telegram/.env")
CHAT_ID = "1738593169"  # Sardor


def _bot_token() -> str:
    token = os.environ.get("CAMERA_AI_BOT_TOKEN")
    if token:
        return token
    try:
        for line in open(TELEGRAM_ENV):
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


_model = None
_model_lock = threading.Lock()


def _pose_model():
    """Pose modeli bitta marta yuklanadi (kameralar o'rtasida bo'lishiladi)."""
    global _model
    with _model_lock:
        if _model is None:
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            m = YOLO(POSE_MODEL)
            m.to(dev)
            m(np.zeros((POSE_IMGSZ, POSE_IMGSZ, 3), dtype=np.uint8),
              device=dev, verbose=False)   # qizdirish
            _model = (m, dev)
        return _model


def raised_hands(frame):
    """Kadrdagi ko'tarilgan qo'llar: [(nom, x, y, odam_bo'yi, ishonch)]"""
    model, dev = _pose_model()
    res = model(frame, conf=POSE_CONF, imgsz=POSE_IMGSZ,
                device=dev, verbose=False)[0]
    return raised_hands_from_result(res)


def raised_hands_from_result(res):
    """Tayyor pose natijasidan ko'tarilgan qo'llarni ajratadi.

    demo.py buni ishlatadi — u pose modelini o'zi bir marta chaqiradi,
    shuning uchun modelni ikkinchi marta yugurtirish shart emas.
    """
    if res.keypoints is None:
        return []

    fh, fw = res.orig_shape[:2]
    out = []
    for kp, box in zip(res.keypoints.data, res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # kadr chetida kesilgan odam: qutisining tepasi haqiqiy tepa emas,
        # shuning uchun "bilak eng tepada" mezoni ishlamaydi -> imo o'qilmaydi
        if x1 <= 3 or y1 <= 3 or x2 >= fw - 3 or y2 >= fh - 3:
            continue
        person_h = y2 - y1
        if person_h < PERSON_MIN_H:    # juda uzoq odam — nuqtalar ishonchsiz
            continue
        k = kp.tolist()
        # haqiqatan odammi? (ryukzak/stulga qarshi)
        if (max(k[L_SH][2], k[R_SH][2]) < SHOULDER_MIN
                or sum(1 for p in k if p[2] >= 0.5) < STRONG_KP_MIN):
            continue
        nose, lsh, rsh = k[NOSE], k[L_SH], k[R_SH]
        # tayanch chiziq = yelkalar. Faqat burunga qarash noto'g'ri edi:
        # qo'l bosh yonida ko'tarilganda burundan pastda qoladi.
        good = [s[1] for s in (lsh, rsh) if s[2] >= KP_CONF]
        if not good:
            continue
        shoulder_y = sum(good) / len(good)

        for name, wr, el in (("chap", k[L_WR], k[L_EL]), ("o'ng", k[R_WR], k[R_EL])):
            if wr[2] < KP_CONF:
                continue
            # 1-shart: bilak yelkadan yuqori
            if wr[1] >= shoulder_y - RAISE_MARGIN * person_h:
                continue
            # 2-shart: bilak odamning eng tepa qismida bo'lsin (boshdan yuqori).
            # Iyagiga/yuziga tiralgan qo'l bu shartdan o'tmaydi.
            if (wr[1] - y1) > TOP_FRAC * person_h:
                continue
            out.append((name, int(wr[0]), int(wr[1]), person_h, float(wr[2])))
    return out


class GestureWatcher:
    """Kadrlarni tekshiradi: kimdir qo'l ko'tarsa Telegram'ga xabar yuboradi."""

    def __init__(self, camera_name: str = "", hold_sec: float = HOLD_SEC,
                 gap_tol: float = GAP_TOL, min_hits: int = 2):
        """hold_sec/gap_tol/min_hits — kamera qanchalik tez tekshirilishiga moslash uchun.

        demo.py da fondagi kamera sekundiga bir marta emas, bir necha sekundda
        bir marta tekshiriladi — u yerda "2 sekund uzluksiz" o'rniga
        "kamida 2 marta ketma-ket ko'rindi" mezoni ishlatiladi.
        """
        self.camera_name = camera_name
        self.hold_sec = hold_sec
        self.gap_tol = gap_tol
        self.min_hits = min_hits
        self.raise_since = 0.0   # qo'l uzluksiz qachondan beri ko'tarilgan
        self.last_seen = 0.0     # oxirgi marta qo'l ko'rilgan vaqt (uzilishga toqat)
        self.hits = 0            # ketma-ket necha tekshiruvda ko'rindi
        self.last_alert = 0.0
        self.last_raised = 0     # UI uchun: hozir nechta qo'l ko'tarilgan
        self.alerts = 0

    def _send(self, frame, count):
        """Telegram'ga rasm bilan xabar (bloklamaslik uchun alohida oqimda)."""
        token = _bot_token()
        if not token:
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        cam = f" — {self.camera_name}" if self.camera_name else ""
        who = "Bir kishi" if count == 1 else f"{count} kishi"
        text = (f"✋ {who} qo'l ko'tardi{cam}\n"
                f"Vaqt: {time.strftime('%H:%M:%S')}")

        def worker():
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": CHAT_ID, "caption": text},
                    files={"photo": ("raise.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=20,
                )
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def check(self, frame, person_boxes=None, draw_on=None) -> int:
        """Kadrni tekshiradi. Qaytaradi: ko'tarilgan qo'llar soni.

        person_boxes kerak emas (pose modeli odamni o'zi topadi), lekin
        eski chaqiruvlar buzilmasin uchun qoldirildi.
        """
        return self.check_hands(raised_hands(frame), frame, draw_on=draw_on)

    def check_hands(self, hands, frame, draw_on=None) -> int:
        """Tayyor topilgan qo'llar bo'yicha signal qoidasini qo'llaydi.

        Ushlab turish (HOLD_SEC), uzilishga toqat (GAP_TOL) va jimlik
        (COOLDOWN_SEC) shu yerda hisoblanadi.
        """
        self.last_raised = len(hands)

        if draw_on is not None:
            for _, x, y, ph, _c in hands:
                r = max(10, int(0.10 * ph))
                cv2.circle(draw_on, (x, y), r, (0, 255, 255), 3)
                cv2.putText(draw_on, "QO'L KO'TARILDI", (x - 90, y - r - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        now = time.time()
        if not hands:
            # bir-ikki kadr tushib qolishi "qo'l tushdi" degani emas
            if now - self.last_seen > self.gap_tol:
                self.raise_since = 0.0
                self.hits = 0
            return 0

        # uzoq tanaffusdan keyin hisobni yangidan boshlaymiz
        if self.raise_since == 0.0 or now - self.last_seen > self.gap_tol:
            self.raise_since = now
            self.hits = 0
        self.last_seen = now
        self.hits += 1

        held = now - self.raise_since
        if (held >= self.hold_sec and self.hits >= self.min_hits
                and now - self.last_alert >= COOLDOWN_SEC):
            self.last_alert = now
            self.alerts += 1
            self.raise_since = 0.0
            self.hits = 0
            self._send(draw_on if draw_on is not None else frame, len(hands))

        return len(hands)


if __name__ == "__main__":
    import sys
    from requests.auth import HTTPDigestAuth

    channel = sys.argv[1] if len(sys.argv) > 1 else "1101"
    s = requests.Session()
    s.auth = HTTPDigestAuth("operator", "0perator1audit")
    url = f"http://192.168.90.251/ISAPI/Streaming/channels/{channel}/picture"
    w = GestureWatcher(f"kanal {channel}")
    print(f"Kanal {channel} — qo'lingizni ko'taring (Ctrl+C — chiqish)")
    while True:
        r = s.get(url, timeout=8)
        f = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        n = w.check(f, draw_on=f)
        held = 0 if w.raise_since == 0 else time.time() - w.raise_since
        print(f"ko'tarilgan qo'l: {n}   ushlab turildi: {held:.1f}s   "
              f"xabarlar: {w.alerts}")
        cv2.imwrite("gesture_debug.jpg", f)
        time.sleep(0.3)
