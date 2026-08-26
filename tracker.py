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


IOU_MATCH = 0.3         # quti oldingi trekка shundan ko'p tegsa — o'sha odam
MISS_TOLERANCE = 8      # trek shuncha kadr topilmasa o'chadi
MISS_SECONDS = 3.0      # yoki shuncha vaqt ko'rinmasa (kadr sekin kelса)


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
        self._next = 1
        self._lock = threading.Lock()

    def update(self, persons, now=None):
        """persons — pose.people_in() natijasi ({box, ...} lar).

        Qaytaradi: har biriga barqaror "tid" (trek raqami) qo'shilgan ro'yxat.
        Son = qaytgan ro'yxat uzunligi (faol treklar).
        """
        now = now or time.time()
        with self._lock:
            unmatched = list(range(len(persons)))
            # Har trekни eng mos yangi quti bilan bog'laymiz (kuchli IoU birinchi)
            pairs = []
            for tid, tr in self._tracks.items():
                for i in unmatched:
                    iou = _iou(tr["box"], persons[i]["box"])
                    if iou >= IOU_MATCH:
                        pairs.append((iou, tid, i))
            pairs.sort(reverse=True)

            taken_tid, taken_i = set(), set()
            for iou, tid, i in pairs:
                if tid in taken_tid or i in taken_i:
                    continue
                taken_tid.add(tid)
                taken_i.add(i)
                tr = self._tracks[tid]
                tr["box"] = persons[i]["box"]
                tr["person"] = persons[i]
                tr["misses"] = 0
                tr["last_seen"] = now
                persons[i]["tid"] = tid

            # Mos kelmagan yangi qutilar — yangi trek
            for i in range(len(persons)):
                if i in taken_i:
                    continue
                tid = self._next
                self._next += 1
                self._tracks[tid] = {"box": persons[i]["box"],
                                     "person": persons[i], "misses": 0,
                                     "last_seen": now}
                persons[i]["tid"] = tid

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
