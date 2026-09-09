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
# Pose modeli bo'g'imlarga tayanadi va shu sababli ORQASI bilan o'tirgan,
# bo'g'imi ko'rinmaydigan odamni o'tkazib yuboradi (o'lchandi: B2 da
# sochi bilan yuzi to'silgan odam topilmadi). Oddiy odam-aniqlash modeli
# esa tana shakliga qaraydi va uni topadi.
#
# Shuning uchun IKKALASI ishlatiladi: pose bo'g'imni ko'rganini topadi,
# detektor qolganini. Natijada B2 da 12 -> 13 (haqiqiy son).
DETECT_MODEL = "yolov8m.pt"
DETECT_CONF = 0.35      # pastroqda soxta topilma ko'payadi (0.25 da 16 ta chiqdi)
# Detektor pose TOPMAGAN odamni qo'shadi — lekin u faqat SHAKLGA qaraydi va
# ba'zan bank/idish/stulni odam deb topadi (o'lchandi: oshxonada silindr
# bankka 1-raqam berdi). Pose bunday xato qilmaydi (bo'g'im kerak). Shuning
# uchun detektor YOLG'IZ qo'shadigan topilma qattiqroq tekshiriladi:
DETECT_ADD_CONF = 0.55      # detektor-only topilma shu ishonchdan yuqori bo'lsin
DETECT_ADD_MIN_H = 120      # va shu bo'ydan katta (mayda obyekt odam emas)
# ── Telefon (3.2) ────────────────────────────────────────────────────
# COCO "cell phone" klassi — o'sha yolov8m o'tishida chiqadi, qo'shimcha
# inference YO'Q. NVR 1080p'da telefon 15-25px, shuning uchun ishonch
# odamnikidan pastroq; yolg'on topilma esa vaqt filtri (detectors) bilan
# kesiladi, bu yerda emas.
PHONE_CLASS = 67
PHONE_CONF = 0.30
PHONE_MIN_PX = 10           # bundan kichik quti shovqin
PHONE_WRIST_MAX = 0.30      # telefon-bilak masofasi bo'yning shu ulushidan kam = qo'lda
PHONE_BOX_PAD = 0.10        # qo'l odam qutisidan biroz chiqishi mumkin (eni ulushi)
# SAHI (bo'lakli) rejimда mayda odamни ATAYIN topamiz — shuning uchun bo'y
# chegarasi past. Soxta topilmalarни statik-filtr (qotган rasm) va ishonch
# ushlaydi.
SAHI_MIN_H = 34
SAHI_SLICES = 3             # ~3x3 bo'lak
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
# Sanash mezoni — DALILLAR YIG'INDISI.
#
# Ilgari yelka ishonchi qat'iy darvoza edi va shu sababli aniq odamlar
# tushib qolardi. Jonli o'lchandi (B2, 2026-08-24): 13 ta ishonchli tana
# nuqtasi va yuz nuqtasi bor topilma yelkasi 0.82 bo'lgani uchun rad
# etilgan; yana biri 12 nuqta bilan 0.75 da rad etilgan. Ikkalasi ham
# haqiqiy odam edi.
#
# Xulosa: NUQTA SONI yelka balidan kuchliroq dalil. Odam turgan burchagiga
# qarab yelkasi to'silishi mumkin, lekin o'nta nuqta tasodifan chiqmaydi.
# Shuning uchun uchta yo'ldan biri yetadi:
COUNT_KP_ALONE = 10         # shuncha nuqta bo'lsa yelka umuman so'ralmaydi
COUNT_SHOULDER_MID = 0.85   # o'rtacha yelka +
COUNT_KP_MID = 7            #   shuncha nuqta
COUNT_SHOULDER_HIGH = 0.93  # juda ishonchli yelka +
COUNT_KP_HIGH = 5           #   kamroq nuqta yetadi

# Holat o'qish (o'tirganmi / boshi pastdami) uchun esa chegara QATTIQ qoladi:
# noto'g'ri sanash bitta raqamni buzadi, noto'g'ri "uxlayapti" degani esa
# odamga jarima yozadi.
POSTURE_SHOULDER_MIN = 0.80
POSTURE_KP_MIN = 6
PERSON_MIN_H = 35       # bundan kichik odamda nuqtalar ishonchsiz

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
_detect = None
_lock = threading.Lock()


def detect_model():
    """Odam-aniqlash modeli (bo'g'imsiz). Pose topolmaganini topadi."""
    global _detect
    with _lock:
        if _detect is None:
            import torch
            from ultralytics import YOLO as _Y
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            m = _Y(DETECT_MODEL)
            m.to(dev)
            m(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), device=dev, verbose=False)
            _detect = (m, dev)
        return _detect


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


_sahi = None


def _sahi_model():
    global _sahi
    with _lock:
        if _sahi is None:
            from sahi import AutoDetectionModel
            _, dev = detect_model()
            _sahi = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=DETECT_MODEL,
                confidence_threshold=DETECT_CONF, device=dev)
        return _sahi


def _sahi_boxes(frame):
    """SAHI bo'lakli aniqlash — kadrни bo'laklarга bo'lib mayda odamни topadi.

    Qaytaradi: [(x1,y1,x2,y2,conf), ...] faqat person.
    """
    from sahi.predict import get_sliced_prediction
    h, w = frame.shape[:2]
    sh = max(320, h // SAHI_SLICES)
    sw = max(320, w // SAHI_SLICES)
    res = get_sliced_prediction(
        frame, _sahi_model(), slice_height=sh, slice_width=sw,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5, verbose=0)
    out = []
    for o in res.object_prediction_list:
        if o.category.name == "person":
            b = o.bbox
            out.append((int(b.minx), int(b.miny), int(b.maxx), int(b.maxy),
                        float(o.score.value)))
    return out


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
        if not (strong >= COUNT_KP_ALONE
                or (shoulder >= COUNT_SHOULDER_MID and strong >= COUNT_KP_MID)
                or (shoulder >= COUNT_SHOULDER_HIGH and strong >= COUNT_KP_HIGH)):
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


# ── Ikki modelni birlashtirish ───────────────────────────────────────
MERGE_IOU = 0.30        # detektor qutisi pose odamiga shuncha tegsa — o'sha odam
MERGE_INSIDE = 0.50


def people_in(frame, sliced=False):
    """Kadrdagi odamlar — pose va odam-detektori birgalikda.

    Avval pose (bo'g'imlar + holat), keyin detektor topgan va pose o'tkazib
    yuborgan odamlar qo'shiladi (holati o'qilmaydi, lekin SANALADI).

    sliced=True — detektor o'rniga SAHI (bo'lakli) ishlatiladi: uzoq/mayda
    odamни ham topadi. Sekinroq, shuning uchun faqat OCHILGAN kamerada.
    """
    found = people(infer(frame))
    phones = []
    if sliced:
        det_boxes = _sahi_boxes(frame)
        min_h = SAHI_MIN_H
        # SAHI odamni topadi, telefonni emas — alohida yengil o'tish
        m, dev = detect_model()
        res = m(frame, conf=PHONE_CONF, imgsz=IMGSZ, classes=[PHONE_CLASS],
                device=dev, verbose=False)[0]
        phones = _phone_boxes(res)
    else:
        m, dev = detect_model()
        # Odam (0) va telefon (67) BITTA o'tishda — min conf telefonniki,
        # odam uchun o'z chegarasi pastda qo'llanadi.
        res = m(frame, conf=min(DETECT_CONF, PHONE_CONF), imgsz=IMGSZ,
                classes=[0, PHONE_CLASS], device=dev, verbose=False)[0]
        det_boxes = [(*map(int, box.xyxy[0]), float(box.conf[0]))
                     for box in res.boxes
                     if int(box.cls[0]) == 0 and float(box.conf[0]) >= DETECT_CONF]
        phones = _phone_boxes(res)
        min_h = DETECT_ADD_MIN_H
    for x1, y1, x2, y2, conf in det_boxes:
        b = (x1, y1, x2, y2)
        # Detektor-only topilma: ishonch va o'lcham gate — pose tasdiqlamagani
        # uchun bank/stul/soyani odam deb qo'shmaslik kerak. Qotган rasm/murol
        # esa keyin static_filter'da chiqariladi.
        if conf < DETECT_ADD_CONF or (y2 - y1) < min_h:
            continue
        if any(_iou(b, p["box"]) >= MERGE_IOU or _inside(b, p["box"]) >= MERGE_INSIDE
               for p in found):
            continue
        found.append({
            "box": b, "height": y2 - y1, "conf": conf,
            "keypoints": None, "strong": 0, "neck": None,
            "reliable": False,        # bo'g'im yo'q — holat o'qilmaydi
            "seated": None, "head_down": None,
            "source": "sahi" if sliced else "detector",
        })
    _attach_phones(found, phones)
    return found


def _phone_boxes(res):
    """YOLO natijasidan telefon qutilari [(x1,y1,x2,y2,conf)]."""
    out = []
    for box in res.boxes:
        if int(box.cls[0]) != PHONE_CLASS or float(box.conf[0]) < PHONE_CONF:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if min(x2 - x1, y2 - y1) < PHONE_MIN_PX:
            continue
        out.append((x1, y1, x2, y2, float(box.conf[0])))
    return out


def _attach_phones(persons, phones):
    """Har telefonni QO'LIDA ushlab turgan odamga bog'laydi: p["phone"].

    Stolda yotgan telefon hodisa emas (3.2 "foydalanish" haqida), shuning
    uchun bog'lanmagan telefon tashlab yuboriladi. Qo'lda ekani:
      - bo'g'imli odam: telefon markazi bilakka yaqin (bo'yga nisbatan);
      - bo'g'imsiz odam (detektor-only): markaz qutining yuqori 60% ida.
    Bir telefon — bitta odam (eng yaqini). Bir odamda ko'pi bilan bitta.
    """
    for p in persons:
        p["phone"] = None
    for x1, y1, x2, y2, conf in phones:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, best_d = None, None
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            pad = (px2 - px1) * PHONE_BOX_PAD
            if not (px1 - pad <= cx <= px2 + pad and py1 <= cy <= py2):
                continue
            h = max(1, p["height"])
            k = p.get("keypoints")
            wrists = [k[i] for i in (L_WR, R_WR)] if k else []
            wrists = [w for w in wrists if w[2] >= KP_CONF]
            if wrists:
                d = min(((w[0] - cx) ** 2 + (w[1] - cy) ** 2) ** 0.5 for w in wrists) / h
                if d > PHONE_WRIST_MAX:
                    continue
            else:
                if cy > py1 + 0.6 * (py2 - py1):
                    continue
                d = 1.0                     # bilak yo'q — zaif bog'lanish
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None and best["phone"] is None:
            best["phone"] = {"box": (x1, y1, x2, y2), "conf": conf,
                             "wrist": None if best_d >= 1.0 else round(best_d, 3)}
