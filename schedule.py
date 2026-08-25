"""Dars jadvali — qaysi xonada, qachon, kim dars o'tadi.

3.1 (darsni o'z vaqtida boshlash), 3.3 (xonani tark etmaslik), 2.1 (kechikish),
2.8 (o'z vaqtida tamomlash) — hammasi shu savolga tayanadi. Jadvalsiz kamera
"mentor xonada yo'q" deyishi mumkin, lekin "dars bo'lishi kerak edi" deya olmaydi.

Manba — Mars API, qo'lda yozilmaydi:

    GET /api/v1/groups?branch_id=N&all_statuses=true

Guruhda: lesson_start_time, lesson_end_time, days, teacher_id, room{name}.
Xona nomi kamera nomiga to'g'ridan-to'g'ri tushadi (A4, B3, A2...) — Yunusobod
NVR sida kanallar aynan shu nomlar bilan atalgan, shuning uchun qo'shimcha
jadval kerak emas. Mos kelmaganlari CAMERA_ALIASES da hal qilinadi.

Mustaqil sinash:
    ./venv/bin/python schedule.py            # bugungi jadval
"""
import os
import json
import time
import threading
import datetime

import requests

import rules   # login/kesh mantiqi u yerda yozilgan, takrorlamaymiz

BASE = "https://api.marsit.uz"
GROUPS_URL = f"{BASE}/api/v1/groups"

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, "schedule_cache.json")

TTL_SEC = 1800          # jadval kunda bir marta o'zgaradi, yarim soat yetadi
_cache = {"at": 0.0, "groups": None}
_lock = threading.Lock()

# Filial nomi -> Mars branch_id. Kamera filiali (nvr.BRANCHES) bilan bog'lash uchun.
# Minor uchun Mars branch_id topilsa qo'shiladi (jadval uchun). Hozircha yo'q.
BRANCH_IDS = {"Yunusobod": 2, "Chilonzor": 4, "Oybek": 17}

# Xona nomi kamera nomiga tushmasa shu yerda tuzatiladi.
# Kalit — xona nomi (Mars), qiymat — kamera nomi (NVR).
CAMERA_ALIASES = {
    "A2 (2)": "A2",
    "Coworking 1": "Coworking",
}

# ── Qoidalarning vaqt chegaralari ────────────────────────────────────
# Matnda aniq yozilganlari (boshqasini o'ylab topmaymiz):
EARLY_MIN = 5           # 2.1: "dars vaqtidan 5 daqiqa oldin ish joyida"
OVERRUN_TOLERANCE = 5   # 2.8: "5 daqiqagacha kechiktirishga ruxsat"
# Matnda raqam yo'q — ehtiyotkor qiymat, yolg'on signal bermaslik uchun:
LATE_START_MIN = 5      # 3.1: darsni shuncha kechiktirib boshlasa
AWAY_MIN = 5            # 3.3: xonadan shuncha vaqt yo'qolsa


def _fetch():
    """Barcha filiallardagi guruhlar. branch_id siz API bo'sh ro'yxat qaytaradi."""
    out = []
    for name, bid in BRANCH_IDS.items():
        page = 1
        while True:
            # rules.authed_get — token keshi va 401 da qayta login shu yerda
            r = rules.authed_get(GROUPS_URL,
                                 params={"branch_id": bid, "page": page,
                                         "all_statuses": True})
            d = r.json()
            groups = d.get("groups") or []
            for g in groups:
                g["_branch"] = name
            out.extend(groups)
            if page >= (d.get("page_count") or 1) or not groups:
                break
            page += 1
    return out


def load(force=False):
    """Guruhlar ro'yxati. API yiqilsa keshdan (jadval kunlab o'zgarmaydi)."""
    with _lock:
        fresh = _cache["groups"] is not None and time.time() - _cache["at"] < TTL_SEC
        if fresh and not force:
            return _cache["groups"]
        try:
            groups = _fetch()
            _cache.update(at=time.time(), groups=groups)
            with open(CACHE_FILE, "w") as f:
                json.dump({"at": time.time(), "groups": groups},
                          f, ensure_ascii=False)
            return groups
        except Exception as e:
            if _cache["groups"] is not None:
                print(f"[schedule] API javob bermadi ({e}) — xotiradagi jadval")
                return _cache["groups"]
            if os.path.exists(CACHE_FILE):
                cached = json.load(open(CACHE_FILE))
                age_h = (time.time() - cached.get("at", 0)) / 3600
                print(f"[schedule] API javob bermadi ({e}) — kesh "
                      f"({age_h:.0f} soat oldingi)")
                _cache.update(at=time.time(), groups=cached["groups"])
                return cached["groups"]
            raise


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.datetime.strptime(value[:8], "%H:%M:%S").time()
    except ValueError:
        return None


def _runs_today(group, day=None):
    """Guruh bugun dars o'tadimi.

    `days`: 1 = toq kunlar, 2 = juft kunlar (Mars GroupDaysType).
    Aniq bilmasak True qaytaramiz — dars bor deb hisoblab tekshirgan yaxshi,
    yo'q deb o'tkazib yuborgandan ko'ra.
    """
    day = day or datetime.date.today()
    started = group.get("date_started")
    finished = group.get("date_finished")
    for value, is_start in ((started, True), (finished, False)):
        if not value:
            continue
        try:
            d = datetime.datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if is_start and day < d:
            return False
        if not is_start and day > d:
            return False

    days = group.get("days")
    if days in (1, 2):
        return day.weekday() % 2 + 1 == days
    return True


def room_name(group):
    room = group.get("room")
    name = (room or {}).get("name") if isinstance(room, dict) else None
    return CAMERA_ALIASES.get(name, name)


def lessons_today(branch=None, day=None):
    """Bugungi darslar: [{room, start, end, teacher_id, group_id, group_name}].

    Xonasi yo'q guruhlar tushib qoladi — kamera ularni baribir bog'lay olmaydi.
    """
    day = day or datetime.date.today()
    out = []
    for g in load():
        if branch and g.get("_branch") != branch:
            continue
        if not _runs_today(g, day):
            continue
        room = room_name(g)
        start = _parse_time(g.get("lesson_start_time"))
        end = _parse_time(g.get("lesson_end_time"))
        if not room or not start or not end:
            continue
        out.append({
            "room": room,
            "start": start,
            "end": end,
            "teacher_id": g.get("teacher_id"),
            "group_id": g.get("id"),
            "group_name": g.get("name"),
            "branch": g.get("_branch"),
        })
    return sorted(out, key=lambda x: (x["room"], x["start"]))


def current_lesson(room, at=None, branch=None):
    """Shu xonada ayni damda ketayotgan dars. Yo'q bo'lsa None."""
    at = at or datetime.datetime.now()
    for lesson in lessons_today(branch=branch, day=at.date()):
        if lesson["room"] != room:
            continue
        if lesson["start"] <= at.time() <= lesson["end"]:
            return lesson
    return None


def upcoming_lesson(room, at=None, branch=None, within_min=30):
    """Shu xonada yaqinda boshlanadigan dars (2.1 kechikish uchun)."""
    at = at or datetime.datetime.now()
    best = None
    for lesson in lessons_today(branch=branch, day=at.date()):
        if lesson["room"] != room:
            continue
        delta = (datetime.datetime.combine(at.date(), lesson["start"]) - at)
        minutes = delta.total_seconds() / 60
        if 0 <= minutes <= within_min and (best is None or minutes < best[1]):
            best = (lesson, minutes)
    return best[0] if best else None


def rooms_with_lessons(branch=None, day=None):
    """Jadvalda uchraydigan xonalar — kamera bilan solishtirish uchun."""
    return sorted({l["room"] for l in lessons_today(branch=branch, day=day)})


def match_cameras(cameras, branch=None):
    """Xona nomlari kamera nomlariga tushdimi.

    cameras — {kanal: nom} (nvr.BRANCH_CAMERAS[filial]).
    Qaytaradi: (mos_kelgan {xona: kanal}, jadvalda_bor_kamerada_yo'q, aksincha)
    """
    by_name = {name: ch for ch, name in cameras.items()}
    rooms = set(rooms_with_lessons(branch=branch))
    matched = {r: by_name[r] for r in rooms if r in by_name}
    missing = sorted(rooms - set(by_name))
    unused = sorted(set(by_name) - rooms)
    return matched, missing, unused


if __name__ == "__main__":
    import nvr

    groups = load(force=True)
    print(f"Bazadan olindi: {len(groups)} ta guruh\n")

    today = datetime.date.today()
    lessons = lessons_today()
    print(f"── Bugun ({today}) {len(lessons)} ta dars")
    for l in lessons:
        print(f"   {l['room']:<12} {l['start']}–{l['end']}  "
              f"{l['group_name']}  (mentor id={l['teacher_id']})")
    print()

    for name in nvr.ENABLED:
        matched, missing, unused = match_cameras(nvr.BRANCH_CAMERAS[name],
                                                 branch=name)
        print(f"── Kamera mosligi ({name} filiali)")
        for room, ch in sorted(matched.items()):
            print(f"   {room:<12} → kanal {ch}")
        if missing:
            print(f"   Jadvalda bor, kamerasi yo'q: {', '.join(missing)}")
        if unused:
            print(f"   Kamerasi bor, bugun darsi yo'q: {', '.join(unused)}")
        print()
