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
# Quti ishonchi ALDAMCHI — odam sanashda unga tayanib bo'lmaydi.
# O'lchandi (B1, deraza oldida o'tirgan odam): quti ishonchi 0.05, ya'ni
# 0.25 chegarada butunlay tushib qolardi. Ayni o'sha topilmaning yelka
# nuqtasi 0.94 va 12 ta kuchli nuqtasi bor edi — ya'ni model odamni ANIQ
# ko'rgan, faqat qutiga past ball bergan.
#
# Shuning uchun: modeldan HAMMASI olinadi (0.05), saralash esa NUQTALAR
# bo'yicha. Eski demo.py da ham shu xulosa yozilgan edi: "odamda yelka
# 0.90-0.98, arvoh qutilarda 0.01-0.15".
CONF = 0.05
KP_CONF = 0.30          # nuqta ishonchi shundan past bo'lsa hisobga olinmaydi
# Ikki xil savol, ikki xil chegara. Ilgari bittasi ishlatilardi va shu sababli
# yarim to'silgan odam umuman sanalmasdi (A2 da ikki kishidan biri tushib
# qolgan edi).
#   SANASH  — bu odammi? (yumshoqroq: stolga yashiringan odam ham odam)
#   HOLAT   — o'tirganmi / boshi pastdami? (qattiq: nuqtalar aniq bo'lsin)
# Sanash mezoni IKKI BOSQICHLI. Yagona "8 ta nuqta" sharti yarim to'silgan
# odamni tashlab yuborardi: B4 da stolga engashgan odamning yelkasi 0.97,
# quti bali 0.74 edi — lekin oyoqlari stol ostida qolgani uchun atigi 7 ta
# nuqtasi ko'rinardi.
#
# Yelka ishonchi eng kuchli dalil, shuning uchun u yuqori bo'lsa kamroq
# nuqta yetadi:
COUNT_SHOULDER_STRONG = 0.90   # yelka shundan yuqori bo'lsa
COUNT_KP_IF_STRONG = 6         # shuncha nuqta yetadi
COUNT_SHOULDER_MIN = 0.85      # yelka bundan past bo'lsa umuman sanalmaydi
COUNT_KP_MIN = 9               # oraliqdagi yelkada esa ko'proq nuqta kerak
POSTURE_SHOULDER_MIN = 0.80  # holat o'qish uchun
POSTURE_KP_MIN = 6
PERSON_MIN_H = 45       # bundan kichik odamda nuqtalar ishonchsiz

# Dublikat qutilar. YOLO bitta odamga ikkita quti berishi mumkin (o'lchandi:
# A2 da bir kishida ikkita, Coworking da chap burchakda beshta ustma-ust).
# Ularning IoU si past bo'lgani uchun modelning o'z NMS i ajratmaydi, lekin
# bo'yin nuqtasi (yelkalar o'rtasi) deyarli bir joyda bo'ladi.
DUP_IOU = 0.55          # qutilar shuncha ustma-ust tushsa — bitta odam
DUP_NECK = 0.30         # yoki bo'yin nuqtalari bo'yning shuncha ulushida yaqin
# Yoki kichik quti kattasining ichida yotsa. IoU buni ushlamaydi: B1 da
# bitta turgan odamga (415,116)-(454,224) va (387,149)-(451,320) qutilari
# tushdi, IoU atigi 0.22 — lekin kichigining 64% i kattasining ichida edi.
DUP_INSIDE = 0.60
# Model bitta odamni yuqori va pastki qismga BO'LIB yuborishi mumkin (A1 da
# stolda o'tirgan odamda shunday bo'ldi). Bunday parchani quti o'lchovlari
# bilan ajratib bo'lmaydi: ikki xil odam (IoU 0.28, ichida 0.51) va bitta
# odamning ikki parchasi (IoU 0.30, ichida 0.47) raqamda deyarli bir xil.
#
# Ajratuvchi belgi — YUZ. Har odamda bitta yuz bor. Parchada burun ham,
# ko'z ham chiqmaydi (o'lchandi: butun qismda burun+ko'z, parchada faqat
# quloq), ikki haqiqiy odamda esa ikkalasida ham yuz nuqtasi bor.
FRAGMENT_INSIDE = 0.40  # shundan ko'p ustma-ust tushsa yuz bo'yicha tekshiramiz

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


def _neck(k):
    """Yelkalar o'rtasi — odamni belgilovchi barqaror nuqta.

    Yelka ko'rinmasa burun, u ham bo'lmasa None.
    """
    pts = [p for p in (k[L_SH], k[R_SH]) if p[2] >= 0.25]
    if pts:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    if k[NOSE][2] >= 0.25:
        return (k[NOSE][0], k[NOSE][1])
    return None


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _face_points(k):
    """Yuz nuqtalari soni (burun, ko'zlar). Quloq hisobga olinmaydi —
    u parchada ham chiqib qoladi."""
    return sum(1 for i in (NOSE, 1, 2) if k[i][2] >= 0.5)


def _inside(a, b):
    """Kichik quti kattasining qancha qismi ichida (0..1)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def _dedupe(cands):
    """Bir odamga tushgan bir necha qutini bittaga yig'adi.

    Kuchliroq nomzod (ishonchli nuqtasi ko'pi, keyin quti ishonchi) qoladi.
    """
    cands = sorted(cands, key=lambda p: (p["strong"], p["conf"]), reverse=True)
    kept = []
    for c in cands:
        dup = False
        for k in kept:
            if _iou(c["box"], k["box"]) >= DUP_IOU:
                dup = True
                break
            inside = _inside(c["box"], k["box"])
            if inside >= DUP_INSIDE:
                dup = True
                break
            # Tana parchasimi: ustma-ust tushgan, lekin yuzi yo'q.
            # Orqasi bilan turgan YOLG'IZ odam bundan zarar ko'rmaydi —
            # u hech kim bilan ustma-ust tushmaydi.
            if (inside >= FRAGMENT_INSIDE
                    and _face_points(c["keypoints"]) == 0
                    and _face_points(k["keypoints"]) > 0):
                dup = True
                break
            if c["neck"] and k["neck"]:
                d = ((c["neck"][0] - k["neck"][0]) ** 2
                     + (c["neck"][1] - k["neck"][1]) ** 2) ** 0.5
                if d <= DUP_NECK * min(c["height"], k["height"]):
                    dup = True
                    break
        if not dup:
            kept.append(c)
    return kept


def people(res):
    """Pose natijasidan odamlar ro'yxati (dublikatlar yig'ilgan).

    Har biri: {box, height, keypoints, seated, head_down, reliable}
    reliable=False — nuqtalar holat o'qish uchun zaif. Bunday odam ham
    SANALADI, faqat "o'tirganmi/uxlayaptimi" degan savolga javob berilmaydi.
    """
    if res.keypoints is None:
        return []

    fh, fw = res.orig_shape[:2]
    cands = []
    for kp, box in zip(res.keypoints.data, res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h = y2 - y1
        k = kp.tolist()
        strong = sum(1 for p in k if p[2] >= 0.5)
        shoulder = max(k[L_SH][2], k[R_SH][2])

        # Odam sifatida sanaladimi. Ikkala shart ham bajarilishi kerak:
        # yolg'iz yelka ishonchi stul suyanchig'ida ham yuqori chiqishi
        # mumkin, yolg'iz nuqta soni esa soyada.
        if h < PERSON_MIN_H:
            continue
        if shoulder < COUNT_SHOULDER_MIN:
            continue
        need = (COUNT_KP_IF_STRONG if shoulder >= COUNT_SHOULDER_STRONG
                else COUNT_KP_MIN)
        if strong < need:
            continue

        # Kadr chetida kesilgan odamning qutisi haqiqiy tana chegarasi emas —
        # bo'yga nisbatan o'lchangan hamma narsa buziladi, holat o'qilmaydi.
        clipped = x1 <= 3 or y1 <= 3 or x2 >= fw - 3 or y2 >= fh - 3
        posture_ok = (not clipped and h >= 80
                      and shoulder >= POSTURE_SHOULDER_MIN
                      and strong >= POSTURE_KP_MIN)

        cands.append({
            "box": (x1, y1, x2, y2),
            "height": h,
            "conf": float(box.conf[0]),
            "keypoints": k,
            "strong": strong,
            "neck": _neck(k),
            "reliable": posture_ok,
        })

    out = []
    for c in _dedupe(cands):
        k, h, y1 = c["keypoints"], c["height"], c["box"][1]
        c["seated"] = _seated(k, h) if c["reliable"] else None
        c["head_down"] = _head_down(k, y1, h) if c["reliable"] else None
        out.append(c)
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
