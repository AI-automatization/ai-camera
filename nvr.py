"""Hikvision NVR — kadr olish infratuzilmasi (bir nechta filial).

Har filialning O'Z NVR si bor, ya'ni o'z tezlik budjeti, o'z qulfi va o'z
kameralari. Shuning uchun hamma narsa Branch ichida — global emas: bitta
filial qulflansa boshqasi ishlayveradi.

Ikki jonli o'lchov shu kodni belgilagan:

  1) NVR qulflanadi (2026-08-04 va 2026-08-24 da ikki marta bo'ldi).
     Hikvision ko'p bir vaqtdagi digest autentifikatsiyani hujum deb biladi
     va akkauntni bloklaydi — o'lchandi: unlockTime=1563 sekund. Rad etilgan
     har bir urinish qulfni YANA uzaytiradi, shuning uchun qulf sezilganda
     butunlay to'xtaymiz va muddatni NVR ning o'zidan so'raymiz.

  2) So'rov soni ALDAMCHI o'lchov. NVR sekundiga 17 ta javob berishi mumkin,
     lekin javoblarning ko'pi AYNAN BIR XIL rasm bo'ladi — snapshot oxirgi
     I-frame dan olinadi, kamera esa GovLength=20 va 20 kadr/sek bilan
     ishlaydi, ya'ni yangi I-frame sekundiga bir marta.

     O'lchandi (Chilonzor, kanal 1201, 2026-08-24) — YANGI kadrlar bo'yicha:
         keyframe yo'q, 1 oqim               1.10 yangi kadr/sek
         keyframe har so'rovda, 1 oqim       2.88
         keyframe har so'rovda, 2 oqim       2.25   (bir-birini to'sadi)
         alohida keyframe ipi + 2 oqim       3.75   <- eng yaxshisi
         alohida keyframe ipi + 3 oqim       3.62
     Sub-oqim (1202) yordam bermaydi: GovLength=50, atigi 1.12 yangi kadr/sek.

     Shuning uchun: ochilgan kamerada ALOHIDA ip requestKeyFrame yuboradi va
     IKKI ip kadr oladi. Bir xil kadr qayta e'lon qilinmaydi (dublikatlar
     "silliq video" illyuziyasini berardi, aslida rasm 1 sekund eski edi).

     Haqiqiy yechim kamera sozlamasida: GovLength ni kamaytirish. Lekin bu
     yozuv sifati va disk sarfiga ta'sir qiladi — Bekzod orqali hal qilinadi.
"""
import os
import re
import time
import threading
from collections import deque

import cv2
import hashlib
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

USER = os.environ.get("NVR_USER", "operator")
PASSWORD = os.environ.get("NVR_PASS", "0perator1audit")

BRANCH_HOSTS = {
    "Yunusobod": "192.168.90.251",
    "Chilonzor": "192.168.68.251",
    "Minor": "89.249.60.238:8080",       # tashqi (NAT), HTTP porti 8080
    # Oybek tashqi manzilda va ofis tarmog'idan ochilmaydi (sinaldi: 8080 yopiq).
    "Oybek": "oybek.marsits.uz:8080",
}

# Snapshot URL naqshi filialga qarab farq qiladi. Lokal NVR'lar to'g'ridan-
# to'g'ri kamera oqimini beradi (Streaming/channels), Minor esa NAT orqali
# ulangan kameralarni proksi qiladi (StreamingProxy). {ch} kanal raqami.
SNAPSHOT_PATH = {
    "Minor": "/ISAPI/ContentMgmt/StreamingProxy/channels/{ch}/picture",
}
DEFAULT_SNAPSHOT = "/ISAPI/Streaming/channels/{ch}/picture"

# Filialga xos parol (ba'zi NVR'larda boshqa). Minor eski parolda ekan.
BRANCH_PASS = {}

BRANCH_CAMERAS = {
    "Yunusobod": {
        "101": "Admin-kassa", "201": "Coworking 1", "301": "A2", "401": "A4",
        "601": "Admin", "901": "A5", "1001": "B2", "1101": "B4",
        "1201": "Coworking", "1301": "Oshxona", "1401": "Admin 2",
        "1501": "B1", "1601": "B3", "1701": "A2 (2)", "1801": "A1",
    },
    "Chilonzor": {
        "101": "Stage-3", "201": "Saturn", "301": "Camera 01", "401": "Jupiter",
        "501": "Earth", "601": "Kitchen", "701": "Stage-3 (2)", "801": "Neptun",
        "901": "Administration", "1001": "Venera", "1101": "Toilet/Library",
        "1201": "Co-Working", "1301": "Co-Working (2)", "1401": "Mercury",
        "1501": "Co-Working (3)", "1601": "Kassa",
    },
    "Oybek": {"101": "Kirish", "201": "Zal", "401": "Koridor", "501": "Xona"},
    # Minor NVR nomlari sozlanmagan (hammasi "Camera 01") — kanal raqami bilan
    "Minor": {"101": "Kamera 1", "201": "Kamera 2", "301": "Kamera 3",
              "401": "Kamera 4", "501": "Kamera 5", "601": "Kamera 6",
              "701": "Kamera 7", "801": "Kamera 8"},
}

# Qaysi filiallar ishga tushadi. Vergul bilan: BRANCHES="Yunusobod,Chilonzor"
ENABLED = [b.strip() for b in
           os.environ.get("BRANCHES", "Yunusobod,Chilonzor,Minor").split(",")
           if b.strip() in BRANCH_HOSTS]

# NVR ba'zan javob bermay qoladi: 60 sekundlik o'lchovda 2.24, 2.07 va 1.05
# sekundlik so'rovlar uchradi. Ikkita oquvchi ip bo'lsa, ikkalasi ham shunday
# so'rovga tiqilib qolsa ekran muzlaydi — Sardor ko'rgan qotish shundan edi.
#
# Yechim: ko'proq ip va QISQA kutish. Tiqilgan so'rov 2 sekunddan keyin
# tashlanadi va qaytadan urinadi, ip esa 10 sekund band bo'lib turmaydi.
# O'lchandi (A4, 40 sekunddan):
#     2 oqim, kutish 10s   6.53 kadr/sek  eng yomon uzilish 0.74s
#     4 oqim, kutish  3s   7.24           2.16s
#     6 oqim, kutish  2s   8.45           0.60s   <- eng yaxshisi
MAX_CONN = 10             # bitta NVR ga bir vaqtda shuncha so'rov
FOCUS_WORKERS = 6         # ochilgan kamerani shuncha oqim bilan tortamiz
FETCH_TIMEOUT = 2.0       # kadr so'rovi shuncha kutadi, keyin qaytadan

# Keyframe majburlash TEZLIKNI oshiradi, lekin SIFATNI tushiradi: kamera
# bir xil bitrate'ni ko'proq I-frame'ga bo'lib beradi va har biri kambag'al
# chiqadi. O'lchandi (B4, 1024 kbps):
#     uzluksiz   9.12 kadr/sek   tiniqlik  865
#     150 ms     5.75            1676        <- standart
#     300 ms     3.19            2111
#     1 sekund   1.50            2578
# B1 da farq yo'q (sahna sodda, kamera baribir kam bit sarflaydi) — Sardor
# aynan shuni sezgan: B1 yaxshi, B4/B2 "bijir-bijir".
KEYFRAME_GAP = float(os.environ.get("KEYFRAME_GAP", "0.15"))
# Grid kameralari MUZLATILGAN: doimiy so'rov yubormaydi, oxirgi kadr turadi.
# Sardorning talabi — 15 kamera birdaniga ishlashi kerak emas, kamera faqat
# bosib kirilganda ishlasin. Bu qotishning ham sababi edi: 15 kamera bir
# vaqtda uyg'onib, ulanish limitini band qilardi va ochilgan kamera 0.7-2.1
# sekundga muzlab qolardi (o'lchangan, Yunusobod B3).
#
# Sanoq esa: ishga tushganda bir marta aylanib chiqiladi, keyin faqat
# so'ralganda (sahifadagi "Sanash" tugmasi) yoki kamera ochilganda.
SCAN_GAP = 0.7            # navbatdagi kameralar orasidagi tanaffus
FOCUS_TTL = 6.0           # brauzer jim qolsa fokus bekor bo'ladi
REACH_TIMEOUT = 4         # filial ulanadimi — shuncha kutamiz


def placeholder(text="Ulanmoqda..."):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (26, 22, 18)
    cv2.putText(img, text, (150, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (150, 150, 150), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


PLACEHOLDER = placeholder()
OFFLINE_JPEG = placeholder("Ulanmadi")


class Branch:
    """Bitta filialning NVR si: kameralari, budjeti, qulfi."""

    local = False        # NVR (tezlik budjeti cheklangan)

    def __init__(self, name, host, cameras):
        self.name = name
        self.host = host
        self.password = BRANCH_PASS.get(name, PASSWORD)
        self.gate = threading.Semaphore(MAX_CONN)
        self.locked_until = 0.0
        self._lock_checked = 0.0
        self.reachable = None          # None = hali tekshirilmagan
        self.focus = {"channel": None, "until": 0.0}
        # Har bir oqim so'roviga navbat raqami beriladi. Faqat ENG OXIRGI
        # so'rov fokusni ushlaydi. Aks holda kamera almashtirilganda eski
        # oqim ham fokusni tortib turadi va ikkisi navbatlashib, ekran
        # qorayib qoladi (logda ko'rindi: 801 va 1601 ketma-ket).
        self.stream_token = 0
        self._scan_wanted = threading.Event()
        self.scanning = False
        self.scanned_at = 0.0
        self.cameras = {ch: Camera(self, ch, nm) for ch, nm in cameras.items()}

    # ── qulf ─────────────────────────────────────────────────────────
    def lock_seconds_left(self):
        """NVR qulfi qolgan vaqti (sekund). Qulf bo'lmasa 0."""
        try:
            r = requests.get(f"http://{self.host}/ISAPI/Security/userCheck",
                             auth=HTTPDigestAuth(USER, self.password), timeout=6)
            if "<lockStatus>lock</lockStatus>" in r.text:
                m = re.search(r"<unlockTime>(\d+)</unlockTime>", r.text)
                return int(m.group(1)) if m else 60
        except Exception:
            pass
        return 0

    def note_lockout(self):
        """401 kelganda: muddatni NVR dan so'rab, shungacha to'xtaymiz.

        Ilgari bu yerda qat'iy 60 sekund yozilgan edi — xato: haqiqiy qulf
        ~26 daqiqa bo'ladi, ya'ni kameralar bir daqiqadan keyin yana urinib
        qulfni qayta boshlatardi.
        """
        now = time.time()
        if now - self._lock_checked < 30:   # tekshiruvning o'zi ham so'rov
            self.locked_until = max(self.locked_until, now + 60)
            return
        self._lock_checked = now
        left = self.lock_seconds_left()
        self.locked_until = now + (left + 15 if left else 120)
        if left:
            print(f"[{self.name}] akkaunt qulflandi — {left} sek "
                  f"({left // 60} daqiqa) kutamiz")

    def lock_state(self):
        left = self.locked_until - time.time()
        return (left > 0, int(max(0, left)))

    # ── ishga tushish ────────────────────────────────────────────────
    def probe(self):
        """Filial ulanadimi va qulflanmaganmi — ishga tushishda bir marta.

        Bitta tekshiruv 15 ta xato so'rovdan arzon: qulflangan NVR ga
        birdaniga 15 kamera urilsa, har biri 401 olib qulfni uzaytiradi.
        """
        try:
            requests.get(f"http://{self.host}/ISAPI/System/deviceInfo",
                         auth=HTTPDigestAuth(USER, self.password),
                         timeout=REACH_TIMEOUT)
            self.reachable = True
        except Exception:
            self.reachable = False
            print(f"[{self.name}] NVR ulanmadi ({self.host}) — "
                  f"bu filial kameralari bo'sh turadi")
            return
        left = self.lock_seconds_left()
        if left:
            self.locked_until = time.time() + left + 15
            print(f"[{self.name}] akkaunt qulflangan — {left} sek "
                  f"({left // 60} daqiqa). Qulf ochilgach o'zi tiklanadi.")

    def start(self):
        self.probe()
        for _ in range(FOCUS_WORKERS):
            threading.Thread(target=self._focus_worker, daemon=True).start()
        threading.Thread(target=self._keyframe_worker, daemon=True).start()
        threading.Thread(target=self._scanner, daemon=True).start()
        self.request_scan()          # ishga tushganda bir marta — grid bo'sh qolmasin

    def _scanner(self):
        """Faqat SO'RALGANDA bir marta aylanib chiqadi (doimiy emas).

        request_scan() bayroq qo'yadi, bu ip ko'rib bajaradi. Aylanish
        davomida kamera ochilsa — darhol to'xtaydi, ochilgan kamera
        muhimroq.
        """
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(USER, self.password)
        while True:
            if not self._scan_wanted.wait(timeout=1.0):
                continue
            self._scan_wanted.clear()
            self.scanning = True
            for cam in list(self.cameras.values()):
                if self.focused_channel():
                    break                # kimdir qarab turibdi — to'xtaymiz
                if self.reachable is False or time.time() < self.locked_until:
                    break
                if cam.fetch_once(sess) == Camera.FAIL:
                    cam.online = False
                time.sleep(SCAN_GAP)
            self.scanning = False
            self.scanned_at = time.time()

    def request_scan(self):
        """Bitta sanash aylanishini so'raydi."""
        self._scan_wanted.set()

    def _keyframe_worker(self):
        """Ochilgan kameradan uzluksiz yangi I-frame so'raydi.

        Busiz snapshot oxirgi I-frame ni qaytaradi va rasm ~1 sekund eski
        bo'ladi: so'rov 4/sek bo'lsa ham YANGI kadr 1.1/sek edi. Bu aynan
        ko'zga tashlanadigan kechikish.

        Nega alohida ip: kadr oluvchi ipning o'zi PUT qilsa, PUT va GET
        navbatlashib bir-birini kutadi (2.88 -> 2.25 yangi kadr/sek).
        """
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(USER, self.password)
        while True:
            ch = self.focused_channel()
            if not ch or time.time() < self.locked_until or self.reachable is False:
                time.sleep(0.25)
                continue
            url = (f"http://{self.host}/ISAPI/Streaming/channels/{ch}"
                   f"/requestKeyFrame")
            try:
                with self.gate:
                    sess.put(url, timeout=FETCH_TIMEOUT)
            except Exception:
                time.sleep(0.2)
                continue
            if KEYFRAME_GAP:
                time.sleep(KEYFRAME_GAP)

    # ── fokus ────────────────────────────────────────────────────────
    def set_focus(self, channel):
        self.focus.update(channel=channel, until=time.time() + FOCUS_TTL)

    def new_stream_token(self):
        """Yangi oqim so'rovi uchun navbat raqami."""
        self.stream_token += 1
        return self.stream_token

    def focused_channel(self):
        return self.focus["channel"] if time.time() < self.focus["until"] else None

    def _focus_worker(self):
        """Ochilgan kamerani tortadigan umumiy oqim.

        Nima uchun umumiy: har kameraga o'z oqimlarini bersak 15x3=45 ip
        bo'lardi va bo'sh turgan 42 tasi GIL ni band qilib, yetkazishni
        15 dan 9 kadr/sekka tushirardi (o'lchandi).
        """
        sess = requests.Session()
        sess.auth = HTTPDigestAuth(USER, self.password)
        while True:
            ch = self.focused_channel()
            cam = self.cameras.get(ch) if ch else None
            if cam is None:
                time.sleep(0.25)
                continue
            result = cam.fetch_once(sess)
            if result == Camera.FAIL:
                time.sleep(0.2)
            elif result == Camera.SAME:
                time.sleep(0.05)   # yangi I-frame hali tayyor emas

    # ── so'rov ───────────────────────────────────────────────────────
    def get(self, sess, url):
        """Xavfsiz so'rov: ulanish limiti + qulflanishni sezish."""
        if time.time() < self.locked_until or self.reachable is False:
            return None
        with self.gate:
            try:
                r = sess.get(url, timeout=FETCH_TIMEOUT)
            except Exception:
                return None      # tiqilib qoldi — qaytadan urinamiz
        if r.status_code == 401:
            self.note_lockout()
            return None
        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
            return r.content
        return None


class Camera:
    """Bitta kanal: kadr oladi, oxirgi topilmalarni ustiga chizib e'lon qiladi.

    Ko'rsatish tezligi (kadr olish) va tahlil tezligi (YOLO/yuz) ataylab
    ajratilgan — tahlil ulgurmasa ham video qotib qolmaydi.
    """

    def __init__(self, branch, channel, name=None):
        self.branch = branch
        self.channel = channel
        self.name = name or channel
        # NVR bergan JPEG AYNAN o'zi saqlanadi. Ilgari kadr dekod qilinib,
        # ramkalar chizilib, qayta kodlanardi — bu ikkinchi marta siqish edi
        # va rasm xiralashardi. Endi ramkalarni brauzer chizadi, rasm esa
        # kameradan qanday chiqqan bo'lsa shundayligicha ko'rsatiladi.
        self.jpeg = None
        self.seq = 0
        self.online = False
        self.state = {}
        self.fps = 0.0
        self.lock = threading.Lock()
        self._raw = None
        self._raw_lock = threading.Lock()
        self._stamps = deque(maxlen=20)
        self._last_digest = None    # bir xil kadrni ikki marta e'lon qilmaymiz
        self._published_at = 0.0    # eski kadr yangisining ustiga chiqmasin
        self.running = True

    @property
    def key(self):
        """Filiallar aralashmasin: kanal raqamlari filiallarda takrorlanadi."""
        return f"{self.branch.name}/{self.channel}"

    @property
    def url(self):
        path = SNAPSHOT_PATH.get(self.branch.name, DEFAULT_SNAPSHOT)
        base = f"http://{self.branch.host}" + path.format(ch=self.channel)
        return base + "?videoResolutionWidth=1920&videoResolutionHeight=1080"

    # fetch_once natijasi. "same" ni "fail" dan ajratish SHART: dublikat
    # kelishi kamera ishlayotganini bildiradi, uni offline deb belgilash xato.
    NEW, SAME, FAIL = "new", "same", "fail"

    def fetch_once(self, sess):
        """Bitta kadr oladi. NEW / SAME / FAIL qaytaradi.

        Ikki filtr:
          * Dublikat — NVR ko'pincha aynan o'sha rasmni qaytaradi. Uni qayta
            dekod qilib, chizib, kodlash bekor mehnat va tezlik ko'rsatkichini
            aldaydi (11 kadr/sek ko'rinardi, aslida 1.1 tasi yangi edi).
          * Tartib — ikki ip parallel so'raydi, sekinroq javob keyinroq
            keladi. Eski kadrni yangisining ustiga qo'ysak video orqaga
            sakraydi. Shuning uchun so'rov BOSHLANGAN vaqt solishtiriladi.
        """
        started = time.time()
        data = self.branch.get(sess, self.url)
        if data is None:
            return self.FAIL
        self.online = True             # javob keldi — kamera ishlayapti
        digest = hashlib.md5(data).digest()
        with self.lock:
            if digest == self._last_digest:
                return self.SAME       # o'sha rasm — dekod ham qilmaymiz
            if started < self._published_at:
                return self.SAME       # bundan yangirog'i allaqachon chiqqan
            self._last_digest = digest
            self._published_at = started
        # Dekod QILINMAYDI. Tahlil kerak bo'lganda analizator o'zi dekod
        # qiladi (sekundiga bir marta), ko'rsatish uchun esa dekod ham,
        # qayta kodlash ham shart emas.
        with self._raw_lock:
            self._raw = data
        self._publish(data)
        return self.NEW

    def _publish(self, data):
        now = time.time()
        with self.lock:
            self.jpeg = data
            self.seq += 1
            self._stamps.append(now)
            if len(self._stamps) > 1:
                span = self._stamps[-1] - self._stamps[0]
                self.fps = (len(self._stamps) - 1) / span if span > 0 else 0.0

    def take_frame(self):
        """Tahlil uchun kadr (dekod qilingan). Bir marta beradi.

        Dekod shu yerda bo'ladi — ko'rsatish yo'lida emas. Analizator
        sekundiga bir marta chaqiradi, ko'rsatish esa 7 marta.
        """
        with self._raw_lock:
            data, self._raw = self._raw, None
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def apply(self, state):
        with self.lock:
            self.state = state

    def snapshot(self):
        with self.lock:
            if self.jpeg:
                return self.jpeg
        return OFFLINE_JPEG if self.branch.reachable is False else PLACEHOLDER


# ── Filiallarni yig'ish ──────────────────────────────────────────────
BRANCHES = {}


def build():
    """ENABLED dagi filiallarni yaratadi va ishga tushiradi."""
    for name in ENABLED:
        BRANCHES[name] = Branch(name, BRANCH_HOSTS[name], BRANCH_CAMERAS[name])
    if os.environ.get("MAC_CAMERA_OFF") != "1":
        BRANCHES["Mac"] = LocalBranch()
    for br in BRANCHES.values():
        br.start()
    return BRANCHES


def all_cameras():
    """{'Filial/kanal': Camera} — barcha filiallardagi kameralar."""
    return {cam.key: cam for br in BRANCHES.values()
            for cam in br.cameras.values()}


def find(branch, channel):
    br = BRANCHES.get(branch)
    return br.cameras.get(channel) if br else None


# ── Mac kamerasi ─────────────────────────────────────────────────────
# Yuz tanishni sinash uchun. NVR kameralarida yuz 13-41 piksel bo'ladi va
# tanib bo'lmaydi (o'lchangan), Mac kamerasida esa yuz katta chiqadi —
# tanish shu yerda haqiqatan sinaladi.
#
# Bu NVR emas, lekin Branch/Camera bilan bir xil interfeysni beradi, shuning
# uchun ilova uni oddiy filial kabi ko'radi.
LOCAL_INDEX = int(os.environ.get("MAC_CAMERA", "0"))
LOCAL_WIDTH, LOCAL_HEIGHT = 1920, 1080
LOCAL_JPEG_QUALITY = 90


class LocalCamera:
    """Kompyuterga ulangan kamera (cv2.VideoCapture)."""

    NEW, SAME, FAIL = "new", "same", "fail"

    def __init__(self, branch, channel="0", name="Mac kamera"):
        self.branch = branch
        self.channel = channel
        self.name = name
        self.jpeg = None
        self.seq = 0
        self.online = False
        self.state = {}
        self.fps = 0.0
        self.lock = threading.Lock()
        self._raw = None
        self._raw_lock = threading.Lock()
        self._stamps = deque(maxlen=20)
        self.running = True

    @property
    def key(self):
        return f"{self.branch.name}/{self.channel}"

    def start(self):
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        """Kamera FAQAT kerak bo'lganda ochiladi.

        Ilgari u ishga tushishda ochilib, doim yoqilib turardi — Mac'da
        kamera chirog'i o'chmasdi. Endi kamera bu filial ochilganda (yoki
        xodim qo'shilayotganda) yoqiladi va IDLE_OFF sekunddan keyin
        o'chadi: kamera yoqiq turishi foydalanuvchiga ko'rinadi va uni
        bekorga yoqib qo'yish to'g'ri emas.
        """
        cap = None
        while self.running:
            if not self.wanted():
                if cap is not None:
                    cap.release()
                    cap = None
                    self.online = False
                    print(f"[{self.branch.name}] kamera o'chirildi")
                time.sleep(0.3)
                continue

            if cap is None:
                cap = cv2.VideoCapture(LOCAL_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, LOCAL_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LOCAL_HEIGHT)
                if not cap.isOpened():
                    print(f"[{self.branch.name}] kamera ochilmadi — Terminal'ga "
                          f"kamera ruxsati kerak "
                          f"(System Settings > Privacy & Security > Camera)")
                    self.branch.reachable = False
                    cap.release()
                    cap = None
                    time.sleep(3)
                    continue
                self.branch.reachable = True
                print(f"[{self.branch.name}] kamera yoqildi")

            ok, frame = cap.read()
            if not ok:
                self.online = False
                time.sleep(0.2)
                continue
            self.online = True
            with self._raw_lock:
                self._raw = frame
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, LOCAL_JPEG_QUALITY])
            if not ok:
                continue
            now = time.time()
            with self.lock:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self._stamps.append(now)
                if len(self._stamps) > 1:
                    span = self._stamps[-1] - self._stamps[0]
                    self.fps = (len(self._stamps) - 1) / span if span > 0 else 0.0
            # 30 kadr/sek kerak emas — protsessorni bo'shatamiz
            time.sleep(0.05)
        if cap is not None:
            cap.release()

    def take_frame(self):
        with self._raw_lock:
            frame, self._raw = self._raw, None
        return frame

    def apply(self, state):
        with self.lock:
            self.state = state

    def snapshot(self):
        with self.lock:
            return self.jpeg or PLACEHOLDER

    def wanted(self):
        """Kamera hozir kerakmi — filial ochilgan yoki yaqinda so'ralgan."""
        return time.time() < self.branch.wanted_until

    def fetch_once(self, sess=None):
        return self.SAME       # o'zi oladi, skaner tegmasin


class LocalBranch:
    """Mac kamerasi uchun soxta filial — Branch bilan bir xil interfeys."""

    local = True         # kompyuterdagi kamera: budjet cheklovi yo'q,
                         # shuning uchun grid'da ham jonli ko'rsatiladi
    IDLE_OFF = 8.0       # so'ralmasa shuncha sekunddan keyin o'chadi

    def __init__(self, name="Mac"):
        self.name = name
        self.host = "local"
        self.reachable = None
        self.locked_until = 0.0
        self.scanning = False
        self.scanned_at = 0.0
        self.stream_token = 0
        self.focus = {"channel": None, "until": 0.0}
        self.wanted_until = 0.0
        self.cameras = {"0": LocalCamera(self)}

    def want(self):
        """Kamera kerak — yoqib turamiz (IDLE_OFF sekundga)."""
        self.wanted_until = time.time() + self.IDLE_OFF

    def start(self):
        for cam in self.cameras.values():
            cam.start()

    def lock_state(self):
        return (False, 0)

    def set_focus(self, channel):
        self.focus.update(channel=channel, until=time.time() + FOCUS_TTL)
        self.want()

    def focused_channel(self):
        return self.focus["channel"] if time.time() < self.focus["until"] else None

    def new_stream_token(self):
        self.stream_token += 1
        return self.stream_token

    def request_scan(self):
        self.want()
        self.scanned_at = time.time()
