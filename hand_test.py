"""Qo'l/barmoq aniqlashni NVR kadrida sinash — 5 barmoq ko'rsatilganini topadimi?

Ishlatish:  ./venv/bin/python hand_test.py [kanal]
Har kadrda topilgan qo'llar va barmoq sonini chop etadi, natijani hand_test.jpg ga yozadi.
"""
import sys
import time

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

NVR_HOST = "192.168.90.251"
NVR_USER = "operator"
NVR_PASS = "0perator1audit"
CHANNEL = sys.argv[1] if len(sys.argv) > 1 else "101"

# barmoq uchi va o'rta bo'g'im indekslari (bosh barmoqdan tashqari)
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]


def count_fingers(lm, handedness: str) -> int:
    """Ko'tarilgan barmoqlar soni. lm — 21 ta normalizatsiyalangan nuqta."""
    up = 0
    for tip, pip in zip(TIPS, PIPS):
        # uchi bo'g'imdan yuqorida (y kichikroq) => barmoq ochiq
        if lm[tip].y < lm[pip].y:
            up += 1
    # bosh barmoq: yon tomonga ochiladi, qo'lning chap/o'ngligiga bog'liq
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            up += 1
    else:
        if lm[4].x > lm[3].x:
            up += 1
    return up


def make_detector():
    return HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=4,            # kadrda bir necha odam bo'lishi mumkin
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    ))


def grab(sess) -> np.ndarray:
    url = f"http://{NVR_HOST}/ISAPI/Streaming/channels/{CHANNEL}/picture"
    r = sess.get(url, timeout=8)
    if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
        return None
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def main():
    det = make_detector()
    sess = requests.Session()
    sess.auth = HTTPDigestAuth(NVR_USER, NVR_PASS)
    print(f"Kanal {CHANNEL} — kameraga qo'l ko'rsating. Ctrl+C bilan to'xtatiladi.\n")

    while True:
        frame = grab(sess)
        if frame is None:
            print("kadr yo'q...")
            time.sleep(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        hands = res.hand_landmarks or []
        if hands:
            parts = []
            for i, lm in enumerate(hands):
                side = res.handedness[i][0].category_name
                n = count_fingers(lm, side)
                parts.append(f"{side}={n}")
                # chizamiz
                h, w = frame.shape[:2]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for x, y in pts:
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts)
                color = (0, 0, 255) if n == 5 else (0, 255, 0)
                cv2.putText(frame, f"{n}", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            print(f"qo'l: {len(hands)}  ->  {', '.join(parts)}"
                  + ("   *** BESHTA! ***" if any(
                      count_fingers(lm, res.handedness[i][0].category_name) == 5
                      for i, lm in enumerate(hands)) else ""))
            cv2.imwrite("hand_test.jpg", frame)
        else:
            print("qo'l topilmadi")

        time.sleep(0.4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nto'xtatildi")
