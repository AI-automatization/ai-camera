"""
Hikvision NVR + YOLO odam aniqlash — jonli web ilova (lokal, API'siz).
Brauzerda ochish: http://localhost:5001
"""
import os
import time
import threading

import cv2
import numpy as np
import torch
import requests
from requests.auth import HTTPDigestAuth
from flask import Flask, Response, render_template_string, request, jsonify
from ultralytics import YOLO

from gesture import GestureWatcher

# ---- Sozlamalar ----
NVR_HOST = "192.168.90.251"
NVR_USER = "operator"
NVR_PASS = "0perator1audit"
RTSP_PORT = 554
MODEL = "yolov8m.pt"
IMGSZ = 736          # kichikroq => tezroq (odam aniqlash uchun yetarli)
# B4 da o'lchandi (2026-08-03): haqiqiy odamlar 0.56-0.89, arvoh qutilar 0.20-0.36.
# 0.20 da stul/ryukzak/soya ham "odam" deb sanalardi.
CONF = 0.45
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU

# ---- Bejik (ko'k lenta) aniqlash ----
# Bejik lentasi to'yingan ko'k: HSV H~110-127, S yuqori (portretdan o'lchangan)
BADGE_LO = (100, 90, 60)     # ko'k pastki chegara (H,S,V)
BADGE_HI = (132, 255, 255)   # ko'k yuqori chegara
BADGE_MIN_RATIO = 0.020      # ko'krak sohasida shu foizdan ko'p ko'k bo'lsa = bejik bor

CAMERAS = {
    "101": "Admin-kassa", "201": "Coworking_01", "301": "A2", "401": "A4",
    "601": "Admin", "901": "A5 Room", "1001": "B2", "1101": "B4",
    "1201": "COWORKING", "1301": "Kitchen and coworking", "1401": "Admin (2)",
    "1501": "B1", "1601": "B3", "1701": "A2 (2)", "1801": "A1",
}

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print(f"Model yuklanyapti... (device={DEVICE})")
model = YOLO(MODEL)
model.to(DEVICE)
# warmup — birinchi inference sekin, oldindan qizdiramiz
model(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), device=DEVICE, verbose=False)
print("Model tayyor.")

app = Flask(__name__)


def loading_jpeg(text="Ulanmoqda..."):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (25, 20, 15)
    cv2.putText(img, text, (430, 370), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (200, 200, 200), 3)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


_LOADING = loading_jpeg()


def rtsp_url(channel: str) -> str:
    return (f"rtsp://{NVR_USER}:{NVR_PASS}@{NVR_HOST}:{RTSP_PORT}"
            f"/Streaming/Channels/{channel}")


def snapshot_url(channel: str) -> str:
    # HTTP snapshot (port 80) — RTSP 554 bloklangan bo'lsa ishlaydi.
    # Rezolyutsiya so'ralmasa NVR 704x576 beradi — qo'l/barmoq uchun juda kam.
    return (f"http://{NVR_HOST}/ISAPI/Streaming/channels/{channel}/picture"
            f"?videoResolutionWidth=1920&videoResolutionHeight=1080")


def has_badge(person_crop) -> bool:
    """Odamning ko'krak/bo'yin sohasida ko'k bejik lentasi bormi?"""
    if person_crop is None or person_crop.size == 0:
        return False
    h, w = person_crop.shape[:2]
    # ko'krak sohasi: bosh ostidan, markazda (lenta shu yerda bo'ladi)
    cy1, cy2 = int(h * 0.15), int(h * 0.60)
    cx1, cx2 = int(w * 0.20), int(w * 0.80)
    chest = person_crop[cy1:cy2, cx1:cx2]
    if chest.size == 0:
        return False
    hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BADGE_LO, BADGE_HI)
    ratio = float((mask > 0).mean())
    return ratio >= BADGE_MIN_RATIO


class Camera:
    """Bitta kanalni o'qib, YOLO bilan annotatsiya qilib turadigan worker."""

    def __init__(self, channel: str):
        self.channel = channel
        self.frame = None          # oxirgi annotatsiyalangan JPEG
        self.count = 0             # oxirgi odamlar soni
        self.no_badge = 0          # bejiksizlar soni
        self.gesture = GestureWatcher(CAMERAS.get(channel, channel))
        self.running = True
        self.lock = threading.Lock()
        self._raw = None           # o'qigich saqlaydigan eng yangi xom kadr
        self._raw_lock = threading.Lock()
        # 1) o'qigich: buferni bo'shatib turadi, doim eng yangi kadr
        threading.Thread(target=self._grab, daemon=True).start()
        # 2) qayta ishlagich: o'z tezligida eng yangi kadrga YOLO qo'llaydi
        threading.Thread(target=self._process, daemon=True).start()

    def _grab(self):
        """HTTP snapshot (port 80) ni uzluksiz so'rab, eng yangi kadrni saqlaydi.
        RTSP (554) bloklangan bo'lsa ham ishlaydi."""
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(NVR_USER, NVR_PASS)
        url = snapshot_url(self.channel)
        while self.running:
            try:
                r = sess.get(url, timeout=8)
                if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                    arr = np.frombuffer(r.content, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._raw_lock:
                            self._raw = frame
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1)
            time.sleep(0.15)  # ~3-5 snapshot/sek

    def _process(self):
        """Eng yangi xom kadrga YOLO qo'llaydi (o'z tezligida, kechikishsiz)."""
        while self.running:
            with self._raw_lock:
                frame = None if self._raw is None else self._raw
                self._raw = None  # bir kadrni ikki marta ishlamaslik
            if frame is None:
                time.sleep(0.01)
                continue

            res = model(frame, classes=[0], conf=CONF, imgsz=IMGSZ,
                        device=DEVICE, verbose=False)[0]
            n = 0
            no_badge = 0
            boxes = []
            clean = frame.copy()   # imo tekshiruvi chiziqlarsiz kadrda ishlasin
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((max(0, x1), max(0, y1), x2, y2))
                # bejikni chizishdan OLDIN toza kesib olamiz
                crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
                badge = has_badge(crop)
                # yashil = bejik bor (normal), ko'k = bejiksiz (BGR: 255,0,0)
                color = (0, 255, 0) if badge else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = "OK" if badge else "BEJIKSIZ"
                cv2.putText(frame, tag, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                n += 1
                if not badge:
                    no_badge += 1

            # qo'l ko'tarish: toza kadrda qidiramiz, topilsa frame ustiga belgilaymiz
            self.gesture.check(clean, boxes, draw_on=frame)

            cv2.rectangle(frame, (10, 10), (420, 55), (0, 0, 0), -1)
            cv2.putText(frame, f"Odamlar: {n}   Bejiksiz: {no_badge}", (20, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if no_badge else (0, 255, 0), 2)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self.lock:
                    self.frame = buf.tobytes()
                    self.count = n
                    self.no_badge = no_badge

    def stop(self):
        self.running = False


# Faol kamera (bitta vaqtda bittasi)
_state = {"cam": None, "channel": "1101"}
_state_lock = threading.Lock()


def clean_channel(channel, fallback):
    """Yaroqsiz/bo'sh kanalni joriy kanalga qaytaradi."""
    if channel in CAMERAS:
        return channel
    return fallback


def get_camera(channel: str, allow_switch: bool) -> Camera:
    """allow_switch=True bo'lsagina kamerani almashtiradi (faqat /stream)."""
    with _state_lock:
        channel = clean_channel(channel, _state["channel"])
        if _state["cam"] is None:
            _state["cam"] = Camera(channel)
            _state["channel"] = channel
        elif allow_switch and channel != _state["channel"]:
            _state["cam"].stop()
            _state["cam"] = Camera(channel)
            _state["channel"] = channel
        return _state["cam"]


PAGE = """
<!doctype html><html lang="uz"><head><meta charset="utf-8">
<title>Kamera AI — odam aniqlash</title>
<style>
  body{margin:0;background:#0f1117;color:#e6e6e6;font-family:system-ui,sans-serif}
  header{padding:14px 20px;background:#161923;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:600}
  select{background:#232735;color:#fff;border:1px solid #333;border-radius:8px;padding:8px 12px;font-size:14px}
  .count{margin-left:auto;font-size:15px;background:#232735;padding:8px 14px;border-radius:8px}
  .count b{color:#4ade80;font-size:18px}
  main{display:flex;justify-content:center;padding:16px}
  img{max-width:100%;border-radius:12px;border:1px solid #232735}
</style></head><body>
<header>
  <h1>🎥 Kamera AI — odam aniqlash</h1>
  <select id="ch" onchange="switchCam()">{{options|safe}}</select>
  <div class="count">Odamlar: <b id="cnt">—</b> &nbsp; Bejiksiz: <b id="nb" style="color:#60a5fa">—</b>
    &nbsp; Qo'l ko'targan: <b id="fg" style="color:#fbbf24">—</b> &nbsp; Signal: <b id="al" style="color:#f87171">0</b></div>
</header>
<main><img id="feed" src="/stream?ch={{channel}}"></main>
<script>
  function switchCam(){
    const ch=document.getElementById('ch').value;
    document.getElementById('feed').src='/stream?ch='+ch+'&t='+Date.now();
  }
  setInterval(async()=>{
    const ch=document.getElementById('ch').value;
    try{const r=await fetch('/count?ch='+ch);const j=await r.json();
      document.getElementById('cnt').textContent=j.count;
      document.getElementById('nb').textContent=j.no_badge;
      document.getElementById('fg').textContent=j.raised;
      document.getElementById('al').textContent=j.alerts;}catch(e){}
  },1000);
</script></body></html>
"""


@app.route("/")
def index():
    channel = request.args.get("ch", "1101")
    opts = "".join(
        f'<option value="{c}"{" selected" if c==channel else ""}>{c} — {name}</option>'
        for c, name in CAMERAS.items()
    )
    return render_template_string(PAGE, options=opts, channel=channel)


@app.route("/stream")
def stream():
    channel = request.args.get("ch", "1101")
    cam = get_camera(channel, allow_switch=True)

    def gen():
        while True:
            with cam.lock:
                frame = cam.frame
            if frame is None:
                # hali ulanmadi — "Ulanmoqda..." ko'rsatamiz (bo'sh ekran emas)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _LOADING + b"\r\n")
                time.sleep(0.3)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/count")
def count():
    channel = request.args.get("ch", "1101")
    cam = get_camera(channel, allow_switch=False)
    return jsonify(count=cam.count, no_badge=cam.no_badge, channel=_state["channel"],
                   raised=cam.gesture.last_raised, alerts=cam.gesture.alerts)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
