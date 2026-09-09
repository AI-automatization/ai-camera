"""Odam kuzatuvi — raqam kadrdan kadrga barqaror qolsin.

Muammo: pose har kadrni qaytadan sanaydi. Bitta odam bir kadrda topilib,
keyingisida (bir lahza to'silса yoki egilса) topilmay qolishi mumkin —
natijada son 3-4-5 orasida jimirlaydi va raqamlar almashadi.

Yechim — oddiy IoU-tracker:
  * har yangi quti oldingi kadr trekларига IoU bo'yicha moslashtiriladi;
  * mos kelса — o'sha trek (o'sha RAQAM) saqlanadi, quti yangilanadi;
  * mos kelmasa — yangi trek (yangi raqam);
  * bir kadr topilmagan trek DARHOL o'chirilmaydi — MISS_TOLERANCE kadr
    kutiladi (odam bir lahza to'silса raqami yo'qolmasin).

Son = faol treklar soni. Bu jimirlashni yo'qotadi: bir kadrlik yo'qolish
trekni o'chirmaydi.

Har kamera uchun alohida Tracker. Holat kamera kalitiga bog'liq.
"""
import time
import threading


IOU_MATCH = 0.2         # quti oldingi trekка shundan ko'p tegsa — o'sha odam
MISS_TOLERANCE = 3      # trek shuncha kadr topilmasa o'chadi
MISS_SECONDS = 1.0      # yoki shuncha vaqt ko'rinmasa. QISQA — aks holda yurgan
                        # odam ortida ARVOH IZI qoladi (eski qutilar 3s turardi)
DIST_FACTOR = 1.0       # markaz masofasi trek o'lchamining shunchasidan kam bo'lsa
                        # — o'sha odam. Yurgan odamни IoU tushса ham ushlab qoladi
                        # (asosiy tuzatish: har qadamда yangi raqam bermaslik)


def _center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def _scale(b):
    return max(1.0, ((b[2] - b[0]) + (b[3] - b[1])) * 0.5)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Bitta kameradagi odamlarni kuzatadi."""

    def __init__(self):
        self._tracks = {}      # id -> {box, misses, last_seen, person}
        self._lock = threading.Lock()

    def _new_id(self):
        """Eng kichik BO'SH raqam. Odam ketganda raqami bo'shaydi va qayta
        ishlatiladi — shuning uchun raqamlar kichik qoladi (1,2,3...), global
        hisoblagichdek "19" ga o'smaydi. Odam ko'ringancha raqami saqlanadi."""
        used = set(self._tracks)
        i = 1
        while i in used:
            i += 1
        return i

    def update(self, persons, now=None):
        """persons — pose.people_in() natijasi ({box, ...} lar).

        Qaytaradi: har biriga barqaror "tid" (trek raqami) qo'shilgan ro'yxat.
        Son = qaytgan ro'yxat uzunligi (faol treklar).
        """
        now = now or time.time()
        with self._lock:
            unmatched = list(range(len(persons)))
            # Har trekни eng mos yangi quti bilan bog'laymiz. Moslash IKKI yo'l:
            # ko'p tegса (IoU) YOKI markazi yaqin (yurgan odam — IoU tushса ham).
            pairs = []
            for tid, tr in self._tracks.items():
                tb = tr["box"]
                dt = max(0.0, now - tr["last_seen"])
                vx, vy = tr.get("vel", (0.0, 0.0))
                sx, sy = vx * dt, vy * dt          # trek shu yergacha siljigan
                pbox = (tb[0]+sx, tb[1]+sy, tb[2]+sx, tb[3]+sy)   # BASHORAT
                tcx, tcy = _center(pbox)
                ts = _scale(tb)
                for i in unmatched:
                    pb = persons[i]["box"]
                    iou = _iou(pbox, pb)           # bashorat bilan tegishmi
                    pcx, pcy = _center(pb)
                    dist = ((tcx - pcx) ** 2 + (tcy - pcy) ** 2) ** 0.5 / ts
                    if iou >= IOU_MATCH or dist <= DIST_FACTOR:
                        score = iou + max(0.0, 1.0 - dist)
                        pairs.append((score, tid, i))
            pairs.sort(reverse=True)

            taken_tid, taken_i = set(), set()
            for score, tid, i in pairs:
                if tid in taken_tid or i in taken_i:
                    continue
                taken_tid.add(tid)
                taken_i.add(i)
                tr = self._tracks[tid]
                ocx, ocy = _center(tr["box"])
                nb = persons[i]["box"]
                ncx, ncy = _center(nb)
                dt = now - tr["last_seen"]
                if dt > 1e-3:
                    ovx, ovy = tr.get("vel", (0.0, 0.0))
                    vx = 0.6 * ovx + 0.4 * (ncx - ocx) / dt
                    vy = 0.6 * ovy + 0.4 * (ncy - ocy) / dt
                    cap = 4.0 * _scale(nb)             # runaway bashoratni cheklaymiz
                    tr["vel"] = (max(-cap, min(cap, vx)), max(-cap, min(cap, vy)))
                tr["box"] = nb
                tr["person"] = persons[i]
                tr["misses"] = 0
                tr["last_seen"] = now
                persons[i]["tid"] = tid

            # Mos kelmagan yangi qutilar — yangi trek
            for i in range(len(persons)):
                if i in taken_i:
                    continue
                tid = self._new_id()
                self._tracks[tid] = {"box": persons[i]["box"],
                                     "person": persons[i], "misses": 0,
                                     "last_seen": now, "vel": (0.0, 0.0)}
                persons[i]["tid"] = tid
                taken_tid.add(tid)    # yangi trek — arvoh sikliga TUSHMASIN
                                      # (aks holda o'sha odam ikki marta chiqadi)

            # Bu kadrda topilmagan treklar — sabr, keyin o'chirish. Sabr
            # ichida bo'lganlar RO'YXATGA QO'SHILADI: odam bir lahza to'silса
            # ham raqami ekranda qoladi ("yo'qolguncha tursin").
            ghosts = []
            for tid in list(self._tracks):
                if tid in taken_tid:
                    continue
                tr = self._tracks[tid]
                tr["misses"] += 1
                if tr["misses"] > MISS_TOLERANCE or now - tr["last_seen"] > MISS_SECONDS:
                    del self._tracks[tid]
                    continue
                ghost = dict(tr["person"])
                ghost["box"] = tr["box"]
                ghost["tid"] = tid
                ghost["ghost"] = True     # topilmadi, oxirgi joyida turibdi
                ghosts.append(ghost)

            return persons + ghosts

    def count(self):
        with self._lock:
            return len(self._tracks)


_trackers = {}
_glob_lock = threading.Lock()


def get(key):
    """Kamera kaliti bo'yicha tracker (yo'q bo'lsa yaratiladi)."""
    with _glob_lock:
        t = _trackers.get(key)
        if t is None:
            t = _trackers[key] = Tracker()
        return t
