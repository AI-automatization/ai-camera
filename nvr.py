"""Hikvision NVR — kadr olish infratuzilmasi.

demo.py dan ajratib olindi (imo-ishora mantiqisiz). Bu yerda faqat "kadrni
qanday olish" masalasi: filiallar, xavfsiz so'rov, kamera oqimi.

Ikki jonli o'lchov shu kodni belgilagan (2026-08-04):

  1) NVR qulflanadi. Hikvision ko'p bir vaqtdagi digest autentifikatsiyani
     hujum deb biladi va akkauntni ~15 daqiqaga bloklaydi (Yunusobodda
     shunday bo'ldi: lockStatus=lock, unlockTime=934). Shuning uchun bitta
     sessiya, ulanish limiti va 401 kelganda to'xtash.

  2) Tezlik budjeti. Snapshot rejimida NVR jami ~9 rasm/sek beradi, RTSP (554)
     yopiq. Shuning uchun budjet taqsimlanadi: ochilgan kamera tez, fondagilar
     sekin — aks holda hammasi qotib ko'rinadi.
"""
import os
import re
import time
import threading

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

MAX_CONN = 4              # NVR ga bir vaqtda shuncha so'rovdan ko'p bo'lmasin
FOCUS_INTERVAL = 0.05     # ochilgan kamera — imkon qadar tez
BG_INTERVAL = 2.0         # hech kim qaramayotganda
BG_SLOW_INTERVAL = 8.0    # kimdir kamera ochib turganda fondagilar
FOCUS_TTL = 6.0           # brauzer jim qolsa fokus bekor bo'ladi

_gate = threading.Semaphore(MAX_CONN)
_locked_until = [0.0]

FOCUS = {"channel": None, "until": 0.0}


def focus(channel):
    FOCUS.update(channel=channel, until=time.time() + FOCUS_TTL)


def focused_channel():
    return FOCUS["channel"] if time.time() < FOCUS["until"] else None


def get(sess, url):
    """NVR ga xavfsiz so'rov: ulanish limiti + qulflanishni sezish.

    401 kelsa NVR akkauntni bloklagan bo'lishi mumkin — qayta urinish qulfni
    faqat uzaytiradi, shuning uchun bir muddat butunlay to'xtaymiz.
    """
    if time.time() < _locked_until[0]:
        return None
    with _gate:
        try:
            r = sess.get(url, timeout=10)
        except Exception:
            return None
    if r.status_code == 401:
        _locked_until[0] = time.time() + 60
        return None
    if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
        return r.content
    return None


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
        self.lock = threading.Lock()
        self._raw = None
        self._raw_lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()

    @property
    def url(self):
        return (f"http://{HOST}/ISAPI/Streaming/channels/{self.channel}"
                f"/picture?videoResolutionWidth=1920&videoResolutionHeight=1080")

    @property
    def keyframe_url(self):
        return f"http://{HOST}/ISAPI/Streaming/channels/{self.channel}/requestKeyFrame"

    def _grab(self):
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(USER, PASSWORD)
        while self.running:
            active = focused_channel()
            is_focused = active == self.channel
            # Snapshot faqat I-frame da olinadi (GOP tufayli har ~2 sekundda).
            # requestKeyFrame uni darhol yaratishga majburlaydi -> real-time
            # bo'ladi (o'lchandi: 2s -> ~0s). Faqat ochilgan kamerada — fonda
            # 2 sekund muhim emas, NVR ni bekorga yuklamaymiz.
            if is_focused:
                request_keyframe(sess, self.keyframe_url)
            data = get(sess, self.url)
            if data is None:
                self.online = False
            else:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.online = True
                    with self._raw_lock:
                        self._raw = frame
                    self._publish(frame)
            time.sleep(FOCUS_INTERVAL if is_focused
                       else (BG_SLOW_INTERVAL if active else BG_INTERVAL))

    def _publish(self, frame):
        with self.lock:
            state = dict(self.state)
        vis = self.on_frame(frame.copy(), state) if self.on_frame else frame
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.seq += 1

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
