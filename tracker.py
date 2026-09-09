"""Odam kuzatuvi — ByteTrack bilan (sanoat standarti).

Ilgari o'zimiz yozgan IoU+velocity tracker bor edi; u tez yurgan odamда
(220px/kadr, past fps) raqamни ALMASHTIRARDI (churn). ByteTrack — Kalman
filtr + Hungarian moslash bilan — 300px/kadrда ham bitta ID ushlaydi
(o'lchandi). Shuning uchun unga o'tildi.

ByteTrack faqat KUZATADI: unga bizning detektsiya (pose+detektor) qutilari
beriladi, u har biriga barqaror ID (tid) qo'yadi. Odam aniqlash — YOLO,
yuz tanish — ArcFace; ByteTrack ularга aloqasiz, alohida ish (sanoq/ID).

Har kamera uchun alohida tracker (get(key)).
"""
import threading
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


def _args():
    # track_buffer — yo'qolgan trek necha kadr saqlanadi (occlusion uchun).
    # Past fps (~5) da 15 kadr ~3s: bir lahza to'silса ID saqlanadi, uzoq
    # ketса o'chadi.
    return SimpleNamespace(
        track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
        track_buffer=15, match_thresh=0.85, fuse_score=True)


class _Det:
    """ByteTrack update() kutgan Results-like: xywh, conf, cls + bool-indeks."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _Det(self.xywh[m], self.conf[m], self.cls[m])


def _to_xywh(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1]


class Tracker:
    """Bitta kameradagi odamlarni ByteTrack bilan kuzatadi."""

    def __init__(self):
        self._bt = BYTETracker(_args())
        self._lock = threading.Lock()

    def update(self, persons, now=None):
        """persons — pose.people_in() natijasi ({box, ...}).

        Har biriga barqaror "tid" qo'shadi. Qaytaradi: kuzatilayotgan
        odamlar ro'yxati (tid bilan). Son = shu ro'yxat uzunligi.
        """
        with self._lock:
            if not persons:
                # bo'sh kadr — ByteTrack holatini yangilash uchun ham chaqiramiz
                self._bt.update(_Det(np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
                return []
            # Bizning odamlar allaqachon "tasdiqlangan" — hammasига yuqori
            # ishonch beramiz (ByteTrack high/low ajratishida high bo'lsin).
            xywh = [_to_xywh(p["box"]) for p in persons]
            conf = [max(0.5, float(p.get("conf", 0.9))) for p in persons]
            cls = [0.0] * len(persons)
            out = self._bt.update(_Det(xywh, conf, cls))
            tracked = []
            for row in out:
                # row: [x1,y1,x2,y2, track_id, conf, cls, det_idx]
                idx = int(row[-1])
                tid = int(row[4])
                if 0 <= idx < len(persons):
                    persons[idx]["tid"] = tid
                    tracked.append(persons[idx])
            return tracked

    def count(self):
        with self._lock:
            return len([t for t in self._bt.tracked_stracks if t.is_activated])


_trackers = {}
_glob_lock = threading.Lock()


def get(key):
    """Kamera kaliti bo'yicha tracker (yo'q bo'lsa yaratiladi)."""
    with _glob_lock:
        t = _trackers.get(key)
        if t is None:
            t = _trackers[key] = Tracker()
        return t
