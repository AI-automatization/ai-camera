"""Kamera monitoring — demo dashboard (klient uchun).

Farqi app.py dan:
  * hamma kamerani BIR VAQTDA, fon rejimida kuzatadi (brauzer ochiq bo'lishi shart emas)
  * bitta ekranda grid, kameraga bosilsa — katta jonli ko'rinish
  * hodisalar tarixi rasmlari bilan saqlanadi
  * bejik aniqlash yo'q (ishonchsiz — demoga kiritilmadi)

Ikki muhim o'lchov shu kodni belgilagan (2026-08-04, jonli o'lchangan):
  1) NVR snapshot rejimida hammasi birga ~9 rasm/sek beradi (bitta so'rov ~1.5s).
     RTSP (554) yopiq. Shuning uchun tezlik BUDJETI taqsimlanadi: ochilgan kamera
     tez so'raladi, fondagilar sekin — aks holda hammasi qotib ko'rinadi.
  2) Odamni quti ishonchi bilan ajratib bo'lmaydi (o'tirgan odam 0.16-0.38,
     arvoh qutilar ham 0.16-0.22). Yelka nuqtasi ajratadi: odamda 0.90-0.98,
     arvohda 0.01-0.15.

Ishga tushirish:
    cd ~/Desktop/camera-ai && ./venv/bin/python demo.py
    brauzer: http://localhost:5050
"""
import os
import json
import time
import threading
from collections import deque

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from flask import Flask, Response, render_template_string, jsonify, abort

import gesture
from gesture import GestureWatcher

# ---- Filiallar ----
# NVR ni qulflab qo'ymaslik uchun: Hikvision juda ko'p bir vaqtdagi digest
# autentifikatsiyani hujum deb biladi va akkauntni ~15 daqiqaga bloklaydi
# (2026-08-04 da Yunusobodda shunday bo'ldi: lockStatus=lock, unlockTime=934).
# Shuning uchun bitta umumiy sessiya + ulanish limiti ishlatiladi.
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
NVR_HOST = os.environ.get("NVR_HOST", BRANCHES[BRANCH]["host"])
NVR_USER = os.environ.get("NVR_USER", "operator")
NVR_PASS = os.environ.get("NVR_PASS", "0perator1audit")
PORT = int(os.environ.get("PORT", "5050"))

MAX_NVR_CONN = 4         # NVR ga bir vaqtda shuncha so'rovdan ko'p bo'lmasin
_nvr_gate = threading.Semaphore(MAX_NVR_CONN)
_locked_until = [0.0]    # NVR qulflansa — shu vaqtgacha umuman so'ramaymiz

# NVR jami ~9 rasm/sek beradi. Bitta kamera yolg'iz ~5 kadr/sek oladi.
# Shuning uchun kimdir kamerani ochsa, fondagilar ataylab sekinlashadi —
# aks holda ular butun kanalni band qilib, ochilgan kamera qotib qoladi.
FOCUS_INTERVAL = 0.05    # ochilgan kamera — imkon qadar tez
BG_INTERVAL = 2.0        # hech kim qaramayotganda (sanoq uchun yetarli)
BG_SLOW_INTERVAL = 8.0   # kimdir kamera ochib turganda fondagilar shu tezlikda
FOCUS_TTL = 6.0          # brauzer jim qolsa, fokus shuncha vaqtdan keyin bekor

POSE_IMGSZ = 736
DET_CONF = 0.15          # past ushlaymiz — ajratishni nuqtalar qiladi
SHOULDER_MIN = 0.80      # odam sanalishi uchun yelka nuqtasi shundan yuqori
STRONG_KP_MIN = 6        # va kamida shuncha ishonchli tana nuqtasi
MAX_EVENTS = 30

CAMERAS = BRANCHES[BRANCH]["cameras"]

app = Flask(__name__)

# ---- Fokus (qaysi kamera ochilgan) ----
FOCUS = {"channel": None, "until": 0.0}


def focused_channel():
    return FOCUS["channel"] if time.time() < FOCUS["until"] else None


# ---- Hodisalar tarixi ----
EVENTS = deque(maxlen=MAX_EVENTS)
_events_lock = threading.Lock()
_event_seq = [0]


def add_event(channel, jpeg, people):
    with _events_lock:
        _event_seq[0] += 1
        EVENTS.appendleft({
            "id": _event_seq[0], "channel": channel,
            "camera": CAMERAS.get(channel, channel),
            "time": time.strftime("%H:%M:%S"), "people": people, "jpeg": jpeg,
        })


def find_event(eid):
    with _events_lock:
        for e in EVENTS:
            if e["id"] == eid:
                return e
    return None


def nvr_get(sess, url):
    """NVR ga xavfsiz so'rov: ulanish limiti + qulflanishni sezish.

    401 kelsa NVR akkauntni bloklagan bo'lishi mumkin — qayta urinish qulfni
    faqat uzaytiradi, shuning uchun bir muddat butunlay to'xtaymiz.
    """
    if time.time() < _locked_until[0]:
        return None
    with _nvr_gate:
        try:
            r = sess.get(url, timeout=10)
        except Exception:
            return None
    if r.status_code == 401:
        _locked_until[0] = time.time() + 60      # nafas olib turamiz
        return None
    if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
        return r.content
    return None


def request_keyframe(sess, url):
    """NVR ni darhol yangi I-frame yaratishga majburlaydi (kechikishga qarshi)."""
    if time.time() < _locked_until[0]:
        return
    try:
        with _nvr_gate:
            sess.put(url, timeout=6)
    except Exception:
        pass


def lock_seconds_left():
    """NVR qulfi qolgan vaqti (sekund). Qulf bo'lmasa 0."""
    try:
        r = requests.get(f"http://{NVR_HOST}/ISAPI/Security/userCheck",
                         auth=HTTPDigestAuth(NVR_USER, NVR_PASS), timeout=6)
        if "<lockStatus>lock</lockStatus>" in r.text:
            import re
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


_PLACEHOLDER = placeholder()


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union else 0.0


def people_and_hands(res):
    """Pose natijasidan haqiqiy odamlarni va ko'tarilgan qo'llarni ajratadi.

    Ikki filtr:
      1) Quti ishonchi yolg'iz yetarli emas — o'tirgan/yarim to'silgan odam ham,
         stul oyog'i ham 0.2 atrofida bo'ladi. Yelka nuqtasi ularni ajratadi.
      2) Bitta odamga ikkita quti tushishi mumkin (o'lchandi: IoU 0.64) — ular
         ustma-ust chizilib ko'zga bitta bo'lib ko'rinadi, lekin ikki marta
         sanaladi. Bo'yin nuqtasi yaqin bo'lsa bitta odam deb hisoblanadi.
    """
    if res.keypoints is None:
        return [], []

    cands = []
    for kp, b in zip(res.keypoints.data, res.boxes):
        k = kp.tolist()
        shoulder = max(k[5][2], k[6][2])
        strong = sum(1 for p in k if p[2] >= 0.5)
        if shoulder < SHOULDER_MIN or strong < STRONG_KP_MIN:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        neck = ((k[5][0] + k[6][0]) / 2, (k[5][1] + k[6][1]) / 2)
        cands.append((float(b.conf[0]), (x1, y1, x2, y2), neck, y2 - y1))

    cands.sort(reverse=True, key=lambda c: c[0])
    boxes, necks, heights = [], [], []
    for _conf, box, neck, ph in cands:
        dup = False
        for i, kept in enumerate(boxes):
            near = (abs(neck[0] - necks[i][0]) ** 2
                    + abs(neck[1] - necks[i][1]) ** 2) ** 0.5
            if _iou(box, kept) > 0.45 or near < 0.35 * min(ph, heights[i]):
                dup = True
                break
        if not dup:
            boxes.append(box)
            necks.append(neck)
            heights.append(ph)
    return boxes, gesture.raised_hands_from_result(res)


def annotate(frame, boxes, hands):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
    for _, x, y, ph, _c in hands:
        r = max(12, int(0.10 * ph))
        cv2.circle(frame, (x, y), r, (0, 200, 255), 3)
        cv2.putText(frame, "QO'L KO'TARILDI", (x - 95, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return frame


class CameraWorker:
    """Bitta kanal: kadr oladi, oxirgi topilmalarni ustiga chizib e'lon qiladi.

    Ko'rsatish tezligi (kadr olish) va aniqlash tezligi (YOLO) ataylab
    ajratilgan — YOLO ulgurmasa ham video qotib qolmaydi, oxirgi qutilar
    yangi kadr ustiga chiziladi.
    """

    def __init__(self, channel):
        self.channel = channel
        self.name = CAMERAS.get(channel, channel)
        self.jpeg = None
        self.seq = 0              # kadr raqami (MJPEG uchun)
        self.count = 0
        self.raised = 0
        self.online = False
        self.alerts = 0
        self.alert_until = 0.0
        # fondagi kamera sekin tekshiriladi -> "2 sekund uzluksiz" o'rniga
        # "kamida 2 marta ketma-ket ko'rindi" mezoni
        self.watcher = GestureWatcher(self.name, hold_sec=1.5,
                                      gap_tol=12.0, min_hits=2)
        self.boxes = []           # oxirgi aniqlangan qutilar (keshda)
        self.hands = []
        self.lock = threading.Lock()
        self._raw = None
        self._raw_lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()

    @property
    def url(self):
        return (f"http://{NVR_HOST}/ISAPI/Streaming/channels/{self.channel}"
                f"/picture?videoResolutionWidth=1920&videoResolutionHeight=1080")

    @property
    def keyframe_url(self):
        return (f"http://{NVR_HOST}/ISAPI/Streaming/channels/{self.channel}"
                f"/requestKeyFrame")

    def _grab(self):
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(NVR_USER, NVR_PASS)
        while self.running:
            active = focused_channel()
            focused = active == self.channel
            # Kechikishni yo'qotish: snapshot faqat I-frame da olinadi, ular
            # esa GOP tufayli har ~2 sekundda keladi. requestKeyFrame NVR ni
            # DARHOL yangi I-frame yaratishga majburlaydi -> snapshot real-time
            # bo'ladi (o'lchandi: 2s -> ~0s). Operator huquqi yetadi.
            # Faqat ochilgan kamerada — fon kameralarda 2s muhim emas, NVR ni
            # bekorga yuklamaslik uchun.
            if focused:
                request_keyframe(sess, self.keyframe_url)
            data = nvr_get(sess, self.url)
            if data is None:
                self.online = False
            else:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.online = True
                    with self._raw_lock:
                        self._raw = frame
                    self._publish(frame)
            if focused:
                time.sleep(FOCUS_INTERVAL)
            else:
                time.sleep(BG_SLOW_INTERVAL if active else BG_INTERVAL)

    def _publish(self, frame):
        """Yangi kadrni oxirgi topilmalar bilan chizib e'lon qiladi."""
        with self.lock:
            boxes, hands = list(self.boxes), list(self.hands)
        vis = annotate(frame.copy(), boxes, hands)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.seq += 1

    def take_frame(self):
        with self._raw_lock:
            frame, self._raw = self._raw, None
        return frame

    def apply(self, boxes, hands):
        with self.lock:
            self.boxes, self.hands = boxes, hands
            self.count, self.raised = len(boxes), len(hands)


WORKERS = {ch: CameraWorker(ch) for ch in CAMERAS}


def focus_helper():
    """Ochilgan kamera uchun qo'shimcha kadr oluvchi.

    O'lchandi: bitta oqim 7.3 kadr/sek, uchta oqim 10.3 kadr/sek beradi —
    NVR parallel so'rovni yaxshi ko'taradi, shuning uchun katta oynadagi
    kamerani bir necha oqim birga so'raydi.
    """
    sess = requests.Session()
    sess.auth = HTTPDigestAuth(NVR_USER, NVR_PASS)
    while True:
        ch = focused_channel()
        if ch is None:
            time.sleep(0.4)
            continue
        w = WORKERS[ch]
        request_keyframe(sess, w.keyframe_url)   # real-time uchun (kechikishga qarshi)
        data = nvr_get(sess, w.url)
        if data is None:
            time.sleep(0.3)
            continue
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            with w._raw_lock:
                w._raw = frame
            w._publish(frame)


def processor():
    """Bitta GPU — kameralarni navbat bilan tekshiradi, ochilgani birinchi."""
    model, dev = gesture._pose_model()
    order = list(CAMERAS)
    i = 0
    while True:
        ch = focused_channel()
        if ch is None or WORKERS[ch]._raw is None:
            ch = order[i % len(order)]
            i += 1
        w = WORKERS[ch]
        frame = w.take_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        res = model(frame, conf=DET_CONF, imgsz=POSE_IMGSZ,
                    device=dev, verbose=False)[0]
        boxes, hands = people_and_hands(res)
        w.apply(boxes, hands)

        vis = annotate(frame.copy(), boxes, hands)   # bir marta chizamiz
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with w.lock:
                w.jpeg = buf.tobytes()
                w.seq += 1

        before = w.watcher.alerts
        w.watcher.check_hands(hands, frame, draw_on=vis)
        if w.watcher.alerts > before:
            w.alerts += 1
            w.alert_until = time.time() + 25
            add_event(ch, cv2.imencode(".jpg", vis,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes(),
                      len(boxes))


# ---------------- Web ----------------

CSS = """
 *{box-sizing:border-box}
 body{margin:0;background:#0b0d12;color:#e8eaf0;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
 a{color:inherit;text-decoration:none}
 header{padding:16px 24px;background:#11141c;border-bottom:1px solid #1e2330;
        display:flex;align-items:center;gap:24px;flex-wrap:wrap;position:sticky;top:0;z-index:5}
 .brand{font-size:17px;font-weight:650}
 .brand span{color:#6b7686;font-weight:400;font-size:13px;display:block;margin-top:2px}
 .kpis{display:flex;gap:12px;margin-left:auto;flex-wrap:wrap}
 .kpi{background:#151924;border:1px solid #1e2330;border-radius:10px;padding:9px 16px;min-width:104px}
 .kpi b{display:block;font-size:22px;line-height:1.15;font-weight:650}
 .kpi small{color:#6b7686;font-size:11px;text-transform:uppercase;letter-spacing:.6px}
 .back{background:#151924;border:1px solid #1e2330;border-radius:9px;padding:8px 14px;font-size:13px}
"""

PAGE = """
<!doctype html><html lang="uz"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kamera monitoring — AI</title><style>""" + CSS + """
 main{display:grid;grid-template-columns:1fr 320px;gap:18px;padding:18px 24px;align-items:start}
 .hero{background:#11141c;border:1px solid #1e2330;border-radius:14px;overflow:hidden;margin-bottom:14px}
 .hero img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#0b0d12}
 .hero .hb{display:flex;align-items:center;gap:10px;padding:11px 15px;font-size:14px}
 .hero .hb .nm{font-weight:650;font-size:15px}
 .hero .hb .live{background:#1c2a1e;color:#7ee08a;border-radius:999px;padding:3px 10px;
                 font-size:11px;letter-spacing:.5px}
 .hero .hb .cnt{margin-left:auto;background:#1a1f2b;border-radius:999px;padding:4px 13px;font-size:13px}
 .hero .hb .cnt b{color:#7ee08a}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
 .tile{background:#11141c;border:1px solid #1e2330;border-radius:10px;overflow:hidden;
       transition:border-color .2s,box-shadow .2s;cursor:pointer}
 .tile.sel{border-color:#3d7dd8}
 .tile:hover{border-color:#2c3446}
 .tile.alert{border-color:#f0801a;box-shadow:0 0 0 1px #f0801a55,0 0 24px #f0801a33}
 .tile.off{opacity:.45}
 .tile img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#0b0d12}
 .bar{display:flex;align-items:center;gap:8px;padding:9px 12px;font-size:13px}
 .bar .nm{font-weight:600}
 .bar .ch{color:#5c6575;font-size:11px}
 .bar .cnt{margin-left:auto;background:#1a1f2b;border-radius:999px;padding:3px 11px;font-size:12px}
 .bar .cnt b{color:#7ee08a}
 .dot{width:7px;height:7px;border-radius:50%;background:#3ba55d}
 .off .dot{background:#5c6575}
 aside{background:#11141c;border:1px solid #1e2330;border-radius:12px;padding:14px;
       position:sticky;top:92px;max-height:calc(100vh - 112px);overflow:auto}
 aside h2{font-size:13px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.7px;color:#6b7686}
 .ev{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #1a1f2b}
 .ev:last-child{border-bottom:0}
 .ev img{width:78px;height:46px;object-fit:cover;border-radius:6px;flex-shrink:0}
 .ev .t{font-size:12px;line-height:1.5}
 .ev .t b{color:#f0a04a;display:block}
 .ev .t span{color:#6b7686}
 .empty{color:#4e5666;font-size:12px;padding:14px 0;text-align:center}
 @media(max-width:900px){main{grid-template-columns:1fr}aside{position:static;max-height:none}}
</style></head><body>
<header>
  <div class="brand">Kamera monitoring — AI aniqlash
    <span>{{branch}} filiali · real vaqtda odam aniqlash va xavfsizlik signali</span></div>
  <div class="kpis">
    <div class="kpi"><b id="k_people">—</b><small>Hozir odam</small></div>
    <div class="kpi"><b id="k_cams">—</b><small>Faol kamera</small></div>
    <div class="kpi"><b id="k_alerts" style="color:#f0a04a">0</b><small>Signal</small></div>
  </div>
</header>
<main>
  <div>
    <div class="hero">
      <img id="hero_img" src="">
      <div class="hb"><span class="live">JONLI</span><span class="nm" id="hero_nm">—</span>
        <span class="cnt">Odam: <b id="hero_cnt">0</b></span></div>
    </div>
    <div class="grid" id="grid"></div>
  </div>
  <aside><h2>Hodisalar</h2><div id="events"><div class="empty">Hozircha hodisa yo'q</div></div></aside>
</main>
<script>
const CAMS = {{cams|safe}};
let SEL = Object.keys(CAMS)[7] || Object.keys(CAMS)[0];   // B4 dan boshlaymiz
const grid = document.getElementById('grid');
for (const [ch, nm] of Object.entries(CAMS)) {
  const d = document.createElement('div');
  d.className = 'tile'; d.id = 'c'+ch;
  d.onclick = () => select(ch);
  d.innerHTML = `<img id="i${ch}" src="/shot/${ch}?t=0" alt="">
    <div class="bar"><span class="dot"></span><span class="nm">${nm}</span>
    <span class="cnt">Odam: <b id="n${ch}">0</b></span></div>`;
  grid.appendChild(d);
}
function select(ch){
  SEL = ch;
  document.getElementById('hero_nm').textContent = CAMS[ch];
  for (const c of Object.keys(CAMS))
    document.getElementById('c'+c).classList.toggle('sel', c===ch);
}
select(SEL);

// Katta oyna: MJPEG o'rniga /shot polling.
// Sabab: MJPEG oqimida TCP buferi to'planib kechikish o'sadi (3+ sekund).
// /shot esa har safar ENG YANGI kadrni beradi — real-time, kechikishsiz.
// Preload usuli: yangi rasm to'liq yuklangach almashtiramiz (miltillamaydi).
const heroImg = document.getElementById('hero_img');
let heroBusy = false;
function heroTick(){
  if (heroBusy) return;
  heroBusy = true;
  const im = new Image();
  im.onload = () => { heroImg.src = im.src; heroBusy = false; };
  im.onerror = () => { heroBusy = false; };
  im.src = '/shot/'+SEL+'?t='+Date.now();
}
setInterval(heroTick, 90);   // ~11 kadr/sek, doim eng yangi
async function tick(){
  try{
    // ochilgan kamera tezlikni olsin deb fokusni ushlab turamiz
    fetch('/api/focus/'+SEL, {method:'POST'});
    const s = await (await fetch('/api/state')).json();
    const hero = s.cameras.find(c=>c.channel===SEL);
    if (hero) document.getElementById('hero_cnt').textContent = hero.count;
    k_people.textContent = s.total_people;
    k_cams.textContent = s.online + '/' + s.total_cams;
    k_alerts.textContent = s.total_alerts;
    for (const c of s.cameras){
      document.getElementById('n'+c.channel).textContent = c.count;
      const t = document.getElementById('c'+c.channel);
      t.classList.toggle('alert', c.alert);
      t.classList.toggle('off', !c.online);
    }
    const ev = document.getElementById('events');
    if (s.events.length){
      ev.innerHTML = s.events.map(e =>
        `<div class="ev"><img src="/event/${e.id}.jpg">
         <div class="t"><b>Qo'l ko'tarildi</b>${e.camera}
         <span>${e.time} · ${e.people} odam</span></div></div>`).join('');
    }
  }catch(e){}
}
function shots(){ const t=Date.now();
  // katta oynadagi kamera MJPEG orqali keladi — uni qayta so'ramaymiz
  for (const ch of Object.keys(CAMS))
    if (ch!==SEL) document.getElementById('i'+ch).src='/shot/'+ch+'?t='+t; }
tick(); setInterval(tick,1200); shots(); setInterval(shots,3000);
</script></body></html>
"""

VIEW = """
<!doctype html><html lang="uz"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{name}} — jonli</title><style>""" + CSS + """
 main{padding:18px 24px;display:flex;justify-content:center}
 img{max-width:100%;border-radius:12px;border:1px solid #1e2330}
</style></head><body>
<header>
  <a class="back" href="/">← Barcha kameralar</a>
  <div class="brand">{{name}}<span>Kanal {{channel}} · jonli</span></div>
  <div class="kpis">
    <div class="kpi"><b id="k_people">—</b><small>Odam</small></div>
    <div class="kpi"><b id="k_raised" style="color:#f0a04a">0</b><small>Qo'l ko'targan</small></div>
    <div class="kpi"><b id="k_alerts" style="color:#f0a04a">0</b><small>Signal</small></div>
  </div>
</header>
<main><img src="/stream/{{channel}}"></main>
<script>
const CH="{{channel}}";
async function tick(){
  try{
    await fetch('/api/focus/'+CH, {method:'POST'});
    const s = await (await fetch('/api/state')).json();
    const c = s.cameras.find(x=>x.channel===CH);
    if(c){ k_people.textContent=c.count; k_raised.textContent=c.raised; k_alerts.textContent=c.alerts; }
  }catch(e){}
}
tick(); setInterval(tick,1500);
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, cams=json.dumps(CAMERAS), branch=BRANCH)


@app.route("/view/<channel>")
def view(channel):
    if channel not in CAMERAS:
        abort(404)
    FOCUS.update(channel=channel, until=time.time() + FOCUS_TTL)
    return render_template_string(VIEW, channel=channel, name=CAMERAS[channel])


@app.route("/api/focus/<channel>", methods=["POST"])
def api_focus(channel):
    if channel not in CAMERAS:
        abort(404)
    FOCUS.update(channel=channel, until=time.time() + FOCUS_TTL)
    return jsonify(ok=True)


@app.route("/stream/<channel>")
def stream(channel):
    w = WORKERS.get(channel)
    if w is None:
        abort(404)

    def gen():
        # MJPEG da klassik tuzoq: server kadrni brauzer ko'rsatgandan tez ishlab
        # chiqarsa, ortiqcha kadrlar TCP buferida to'planib kechikish o'sadi
        # (o'lchandi: 3 sekund orqada). Yechim: chiqarishni ~15 kadr/sekga
        # cheklab, HAR SAFAR eng yangi kadrni berish — oradagi eskilar tashlanadi.
        target_dt = 1.0 / 15
        last = -1
        while True:
            t0 = time.time()
            with w.lock:
                jpeg, seq = w.jpeg, w.seq
            if jpeg is None:
                jpeg = _PLACEHOLDER
            elif seq == last:
                time.sleep(0.01)      # yangi kadr yo'q — biroz kutamiz
                continue
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            dt = time.time() - t0
            if dt < target_dt:        # tezlikni ushlab, buferni bo'sh tutamiz
                time.sleep(target_dt - dt)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shot/<channel>")
def shot(channel):
    w = WORKERS.get(channel)
    if w is None:
        abort(404)
    with w.lock:
        data = w.jpeg or _PLACEHOLDER
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/event/<int:eid>.jpg")
def event_shot(eid):
    e = find_event(eid)
    if e is None:
        abort(404)
    return Response(e["jpeg"], mimetype="image/jpeg")


@app.route("/api/state")
def state():
    now = time.time()
    cams, total, online, alerts = [], 0, 0, 0
    for ch, w in WORKERS.items():
        with w.lock:
            cnt, raised, jpeg = w.count, w.raised, w.jpeg
        total += cnt
        online += 1 if (w.online and jpeg is not None) else 0
        alerts += w.alerts
        cams.append({"channel": ch, "name": w.name, "count": cnt, "raised": raised,
                     "alerts": w.alerts, "online": bool(w.online and jpeg is not None),
                     "alert": now < w.alert_until})
    with _events_lock:
        evs = [{k: v for k, v in e.items() if k != "jpeg"} for e in EVENTS]
    return jsonify(cameras=cams, total_people=total, online=online,
                   total_cams=len(WORKERS), total_alerts=alerts, events=evs[:12])


if __name__ == "__main__":
    print("Model yuklanyapti...")
    gesture._pose_model()
    threading.Thread(target=processor, daemon=True).start()
    for _ in range(2):      # ochilgan kamera uchun qo'shimcha kadr oluvchi
        threading.Thread(target=focus_helper, daemon=True).start()
    print(f"Tayyor. Dashboard: http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
