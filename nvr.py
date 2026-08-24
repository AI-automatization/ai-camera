"""Hikvision NVR — kadr olish infratuzilmasi.

demo.py dan ajratib olindi (imo-ishora mantiqisiz). Bu yerda faqat "kadrni
qanday olish" masalasi: filiallar, xavfsiz so'rov, kamera oqimi.

Ikki jonli o'lchov shu kodni belgilagan (2026-08-04):

  1) NVR qulflanadi. Hikvision ko'p bir vaqtdagi digest autentifikatsiyani
     hujum deb biladi va akkauntni ~15 daqiqaga bloklaydi (Yunusobodda
     shunday bo'ldi: lockStatus=lock, unlockTime=934). Shuning uchun bitta
     sessiya, ulanish limiti va 401 kelganda to'xtash.

  2) Tezlik budjeti. RTSP (554) yopiq, faqat snapshot bor. NVR ~17 kadr/sek
     beradi, lekin BITTA oqim 7 kadrdan oshmaydi — shuning uchun ochilgan
     kamera parallel oqimlar bilan tortiladi, fondagilar esa sekin. Aks holda
     15 kamera budjetni bo'lib olib, hammasi qotib ko'rinadi.
"""
import os
import re
import time
import threading

from collections import deque

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

# ── Filiallar ────────────────────────────────────────────────────────
BRANCHES = {
    "Yunusobod": {
        "host": "192.168.90.251",
        "cameras": {
            "101": "Admin-kassa", "201": "Coworking 1", "301": "A2", "401": "A4",
            "601": "Admin", "901": "A5", "1001": "B2", "1101": "B4",
            "1201": "Coworking", "1301": "Oshxona", "1401": "Admin 2",
            "1501": "B1", "1601": "B3", "1701": "A2 (2)", "1801": "A1",
        },
    },
    "Oybek": {
        "host": "oybek.marsits.uz:8080",
        "cameras": {"101": "Kirish", "201": "Zal", "401": "Koridor", "501": "Xona"},
    },
    "Chilonzor": {
        "host": "192.168.68.251",
        "cameras": {
            "101": "Stage-3", "201": "Saturn", "301": "Camera 01", "401": "Jupiter",
            "501": "Earth", "601": "Kitchen", "701": "Stage-3 (2)", "801": "Neptun",
            "901": "Administration", "1001": "Venera", "1101": "Toilet/Library",
            "1201": "Co-Working", "1301": "Co-Working (2)", "1401": "Mercury",
            "1501": "Co-Working (3)", "1601": "Kassa",
        },
    },
}

BRANCH = os.environ.get("BRANCH", "Oybek")
HOST = os.environ.get("NVR_HOST", BRANCHES[BRANCH]["host"])
USER = os.environ.get("NVR_USER", "operator")
PASSWORD = os.environ.get("NVR_PASS", "0perator1audit")
CAMERAS = BRANCHES[BRANCH]["cameras"]

# ── Tezlik budjeti ───────────────────────────────────────────────────
# Jonli o'lchandi (Yunusobod, 2026-08-24, kanal 1601):
#     1 oqim          7.1 kadr/sek
#     2 oqim          12.4
#     3 oqim          16.9   <- to'yinish nuqtasi
#     4 oqim          16.5   (foyda yo'q, faqat ulanish sarflaydi)
# Ya'ni NVR ~17 kadr/sek beradi. Eskirgan izohdagi "~9 kadr/sek" xato edi.
#
# Ochilgan kameraga PARALLEL oqim beriladi — bitta oqim 7 kadrdan oshmaydi,
# uchtasi esa 17 gacha chiqadi. Fondagilar sekin so'raladi: ular faqat odam
# sanash uchun kerak, sekundiga bir marta yangilanishi shart emas.
MAX_CONN = 6              # NVR ga bir vaqtda shuncha so'rov (3 fokus + fon)
FOCUS_WORKERS = 3         # ochilgan kamerani shuncha oqim bilan tortamiz
# Odam sanash — asosiy funksiya, shuning uchun fon kameralari kamera ochilganda
# ham SEKINLASHMAYDI. 15 kamera 10 sekundda = 1.5 so'rov/sek, budjet ~17 —
# ochilgan kameraga qolgani yetib ortadi.
BG_INTERVAL = 10.0
FOCUS_TTL = 6.0           # brauzer jim qolsa fokus bekor bo'ladi

_gate = threading.Semaphore(MAX_CONN)
_locked_until = [0.0]

FOCUS = {"channel": None, "until": 0.0}
REGISTRY = {}             # kanal -> Camera (fokus hovuzi shundan topadi)


def focus(channel):
    FOCUS.update(channel=channel, until=time.time() + FOCUS_TTL)


def focused_channel():
    return FOCUS["channel"] if time.time() < FOCUS["until"] else None


def get(sess, url):
    """NVR ga xavfsiz so'rov: ulanish limiti + qulflanishni sezish.

    401 kelsa NVR akkauntni bloklagan bo'ladi. Qayta urinish qulfni faqat
    uzaytiradi, shuning uchun butunlay to'xtaymiz — va NVR ning O'ZIDAN
    qulf qachon ochilishini so'raymiz.

    Ilgari bu yerda qat'iy 60 sekund yozilgan edi. Bu xato: haqiqiy qulf
    ~15-26 daqiqa bo'ladi (o'lchandi: unlockTime=1563), ya'ni 60 sekunddan
    keyin 15 kamera yana urinib, qulfni qayta boshlatardi.
    """
    if time.time() < _locked_until[0]:
        return None
    with _gate:
        try:
            r = sess.get(url, timeout=10)
        except Exception:
            return None
    if r.status_code == 401:
        note_lockout()
        return None
    if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
        return r.content
    return None


_lock_check = [0.0]


def note_lockout():
    """401 kelganda: NVR dan qulf muddatini so'rab, shungacha to'xtaymiz."""
    now = time.time()
    if now - _lock_check[0] < 30:      # tekshiruvning o'zi ham so'rov — kamdan-kam
        _locked_until[0] = max(_locked_until[0], now + 60)
        return
    _lock_check[0] = now
    left = lock_seconds_left()
    _locked_until[0] = now + (left + 15 if left else 120)
    if left:
        print(f"[nvr] akkaunt qulflandi — {left} sekunddan keyin qayta urinamiz")


def lock_state():
    """(qulflanganmi, qolgan sekund) — dashboardda ko'rsatish uchun."""
    left = _locked_until[0] - time.time()
    return (left > 0, int(max(0, left)))


def request_keyframe(sess, url):
    """NVR ni darhol yangi I-frame yaratishga majburlaydi (kechikishga qarshi)."""
    if time.time() < _locked_until[0]:
        return
    try:
        with _gate:
            sess.put(url, timeout=6)
    except Exception:
        pass


def lock_seconds_left():
    """NVR qulfi qolgan vaqti (sekund). Qulf bo'lmasa 0."""
    try:
        r = requests.get(f"http://{HOST}/ISAPI/Security/userCheck",
                         auth=HTTPDigestAuth(USER, PASSWORD), timeout=6)
        if "<lockStatus>lock</lockStatus>" in r.text:
            m = re.search(r"<unlockTime>(\d+)</unlockTime>", r.text)
            return int(m.group(1)) if m else 60
    except Exception:
        pass
    return 0


def placeholder(text="Ulanmoqda..."):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (26, 22, 18)
    cv2.putText(img, text, (170, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (150, 150, 150), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


PLACEHOLDER = placeholder()


class Camera:
    """Bitta kanal: kadr oladi, oxirgi topilmalarni ustiga chizib e'lon qiladi.

    Ko'rsatish tezligi (kadr olish) va tahlil tezligi (YOLO/yuz) ataylab
    ajratilgan — tahlil ulgurmasa ham video qotib qolmaydi, oxirgi natija
    yangi kadr ustiga chiziladi.
    """

    def __init__(self, channel, on_frame=None):
        self.channel = channel
        self.name = CAMERAS.get(channel, channel)
        self.on_frame = on_frame     # annotatsiya chizuvchi: f(frame, state) -> frame
        self.jpeg = None
        self.seq = 0
        self.online = False
        self.state = {}              # tahlil natijasi (detektorlar to'ldiradi)
        self.fps = 0.0
        self.lock = threading.Lock()
        self._raw = None
        self._raw_lock = threading.Lock()
        self._stamps = deque(maxlen=20)   # oxirgi kadr vaqtlari (fps uchun)
        self.running = True
        # Har bir kamerada BITTA fon oquvchi. Ochilgan kameraga qo'shimcha
        # tezlikni umumiy hovuz beradi (_focus_pool) — har kameraga o'z
        # oquvchilarini berish 15x3=45 ta ip hosil qilardi va ularning 42 tasi
        # bekorga uyg'onib, GIL ni band qilardi (o'lchandi: 15 kadr ishlab
        # chiqarilib, brauzerga 9 tasi yetardi).
        threading.Thread(target=self._grab, daemon=True).start()
        REGISTRY[channel] = self

    @property
    def url(self):
        return (f"http://{HOST}/ISAPI/Streaming/channels/{self.channel}"
                f"/picture?videoResolutionWidth=1920&videoResolutionHeight=1080")

    @property
    def keyframe_url(self):
        return f"http://{HOST}/ISAPI/Streaming/channels/{self.channel}/requestKeyFrame"

    def fetch_once(self, sess):
        """Bitta kadr olib, e'lon qiladi. True — muvaffaqiyat."""
        data = get(sess, self.url)
        if data is None:
            return False
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        self.online = True
        with self._raw_lock:
            self._raw = frame
        self._publish(frame)
        return True

    def _grab(self):
        """Fon oquvchi: kamera ochilmagan bo'lsa ham odam sanashga kadr beradi.

        requestKeyFrame ATAYLAB olib tashlandi: o'lchovda u tezlikni
        7.1 -> 6.1 kadr/sek ga tushirdi (har kadrga qo'shimcha PUT so'rovi).
        Uzluksiz tortganda kechikish baribir kichik — keyingi kadr ~0.15
        sekunddan keyin keladi.
        """
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(USER, PASSWORD)
        while self.running:
            active = focused_channel()
            if active == self.channel:
                # Ochilgan kamerani hovuz tortyapti — bu ip aralashmasin
                time.sleep(0.5)
                continue
            if not self.fetch_once(sess):
                self.online = False
            time.sleep(BG_INTERVAL)

    def _publish(self, frame):
        with self.lock:
            state = dict(self.state)
        vis = self.on_frame(frame.copy(), state) if self.on_frame else frame
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        now = time.time()
        with self.lock:
            self.jpeg = buf.tobytes()
            self.seq += 1
            self._stamps.append(now)
            if len(self._stamps) > 1:
                span = self._stamps[-1] - self._stamps[0]
                self.fps = (len(self._stamps) - 1) / span if span > 0 else 0.0

    def take_frame(self):
        """Tahlil uchun xom kadr. Bir marta beradi — takror tahlil qilinmasin."""
        with self._raw_lock:
            frame, self._raw = self._raw, None
        return frame

    def apply(self, state):
        with self.lock:
            self.state = state

    def snapshot(self):
        with self.lock:
            return self.jpeg or PLACEHOLDER


# ── Fokus hovuzi ─────────────────────────────────────────────────────
# Ochilgan kamerani tortadigan umumiy oqimlar. Ular qaysi kamera ochilganini
# har safar tekshiradi, shuning uchun kamera almashsa ham qo'shimcha ip
# yaratilmaydi. Nima uchun umumiy: har kameraga o'z oqimlarini bersak
# 15x3=45 ip bo'lardi va bo'sh turgan 42 tasi GIL ni band qilib, yetkazish
# tezligini 15 dan 9 kadr/sekka tushirardi.
def _focus_pool_worker():
    sess = requests.Session()
    sess.auth = HTTPDigestAuth(USER, PASSWORD)
    while True:
        ch = focused_channel()
        cam = REGISTRY.get(ch) if ch else None
        if cam is None:
            time.sleep(0.25)
            continue
        if not cam.fetch_once(sess):
            time.sleep(0.1)


def start_focus_pool(workers=FOCUS_WORKERS):
    for _ in range(workers):
        threading.Thread(target=_focus_pool_worker, daemon=True).start()
