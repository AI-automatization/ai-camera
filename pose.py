"""Tana holati — odam turibdimi, o'tiribdimi, boshi partadami.

2-darajali qoidalar shunga tayanadi:
  2.10  mentor darsni o'tirib o'tmasin
  3.10  dars vaqtida uxlash (bosh partada)
  3.11  coworkingda uxlash

YOLOv8-pose bitta o'tishda odamni ham, 17 ta tana nuqtasini ham beradi —
odam aniqlash uchun alohida model kerak emas.

Chegaralar B2/B4/A5 kameralarida jonli o'lchangan (2026-08-03). Shift
kamerasida odam ~80-300 piksel bo'ladi, shuning uchun barcha masofalar
odam bo'yiga nisbatan (piksel emas) — kamera balandligi ta'sir qilmasin.

Mustaqil sinash:
    ./venv/bin/python pose.py rasm.jpg
"""
import threading

import numpy as np
from ultralytics import YOLO

MODEL = "yolov8s-pose.pt"
IMGSZ = 960
CONF = 0.25
KP_CONF = 0.30          # nuqta ishonchi shundan past bo'lsa hisobga olinmaydi
SHOULDER_MIN = 0.80     # yelka ishonchi — ryukzak/stulni odam deb o'qimaslik uchun
STRONG_KP_MIN = 6       # kamida shuncha ishonchli nuqta
PERSON_MIN_H = 80       # bundan kichik odamda holat o'qish ishonchsiz

# COCO-17 nuqta indekslari
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

# ── Holat chegaralari ────────────────────────────────────────────────
# Turgan odamda yelka-son masofasi bo'yning ~0.30 qismi, o'tirganda torayadi
# (kamera tepadan qaraydi — perspektiva siqadi).
SEATED_TORSO_MAX = 0.22
# Uxlayotgan odamda bosh son sathiga yaqinlashadi yoki pastga tushadi.
SLUMPED_HEAD_MIN = 0.55   # bosh y = quti tepasidan shu ulushdan past
# Bir marta engashgan odam uxlagan emas — holat shu vaqt turishi kerak.
SLEEP_HOLD_SEC = 60.0

_model = None
_lock = threading.Lock()


def model():
    """Pose modeli bitta marta yuklanadi (kameralar o'rtasida bo'lishiladi)."""
    global _model
    with _lock:
        if _model is None:
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            m = YOLO(MODEL)
            m.to(dev)
            m(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), device=dev, verbose=False)
            _model = (m, dev)
        return _model


def infer(frame):
    """Kadrni modeldan o'tkazadi. Natijani people() ga berish kerak."""
    m, dev = model()
    return m(frame, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)[0]


def people(res):
    """Pose natijasidan odamlar ro'yxati.

    Har biri: {box, height, keypoints, seated, head_down, reliable}
    reliable=False — nuqtalar zaif, holatga qarab qaror qilinmasin
    (odam uzoq, kadr chetida kesilgan yoki bu umuman odam emas).
    """
    if res.keypoints is None:
        return []

    fh, fw = res.orig_shape[:2]
    out = []
    for kp, box in zip(res.keypoints.data, res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h = y2 - y1
        k = kp.tolist()

        # Kadr chetida kesilgan odamning qutisi haqiqiy tana chegarasi emas —
        # bo'yga nisbatan o'lchangan hamma narsa buziladi.
        clipped = x1 <= 3 or y1 <= 3 or x2 >= fw - 3 or y2 >= fh - 3
        strong = sum(1 for p in k if p[2] >= 0.5)
        reliable = (
            not clipped
            and h >= PERSON_MIN_H
            and max(k[L_SH][2], k[R_SH][2]) >= SHOULDER_MIN
            and strong >= STRONG_KP_MIN
        )

        out.append({
            "box": (x1, y1, x2, y2),
            "height": h,
            "conf": float(box.conf[0]),
            "keypoints": k,
            "reliable": reliable,
            "seated": _seated(k, h) if reliable else None,
            "head_down": _head_down(k, y1, h) if reliable else None,
        })
    return out


def _avg_y(points):
    good = [p[1] for p in points if p[2] >= KP_CONF]
    return sum(good) / len(good) if good else None


def _seated(k, h):
    """O'tirganmi. Yelka bilan son orasidagi masofa torayadi.

    None — hukm qilib bo'lmadi (son nuqtalari ko'rinmayapti).
    """
    sh_y = _avg_y([k[L_SH], k[R_SH]])
    hip_y = _avg_y([k[L_HIP], k[R_HIP]])
    if sh_y is None or hip_y is None or h <= 0:
        return None
    return (hip_y - sh_y) / h < SEATED_TORSO_MAX


def _head_down(k, y1, h):
    """Bosh pastda — partaga qo'yilgan yoki juda engashgan.

    Burun ko'rinmasa quloqqa tushamiz: partaga bosh qo'yganda yuz pastga
    qaraydi va burun ko'pincha yo'qoladi, quloq esa qoladi.
    """
    head_y = _avg_y([k[NOSE]]) or _avg_y([k[L_EAR], k[R_EAR]])
    if head_y is None or h <= 0:
        return None
    return (head_y - y1) / h > SLUMPED_HEAD_MIN


def annotate(frame, persons):
    """Odamlarni holati bilan chizadi."""
    import cv2
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        if not p["reliable"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
            continue
        marks = []
        if p["seated"]:
            marks.append("o'tirgan")
        if p["head_down"]:
            marks.append("bosh pastda")
        color = (0, 0, 255) if p["head_down"] else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if marks:
            cv2.putText(frame, ", ".join(marks), (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        raise SystemExit("foydalanish: python pose.py rasm.jpg")
    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"rasm o'qilmadi: {sys.argv[1]}")

    persons = people(infer(img))
    print(f"{len(persons)} ta odam:")
    for i, p in enumerate(persons, 1):
        if not p["reliable"]:
            print(f"  {i}. (ishonchsiz — bo'y {p['height']}px)")
            continue
        print(f"  {i}. bo'y={p['height']}px  o'tirgan={p['seated']}  "
              f"bosh_pastda={p['head_down']}")
