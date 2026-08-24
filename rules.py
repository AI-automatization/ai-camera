"""MARS audit qoidalari — manbasi Mars API, kamera shu ro'yxatga tayanadi.

Qoidalar bu yerda qo'lda yozilmaydi. Yagona manba:

    GET https://api.marsit.uz/api/v2/reports/audit-rules

Baza o'zgarsa (metodist adminda qoida qo'shsa/o'chirsa) kamera ham o'sha zahoti
yangi ro'yxat bilan ishlaydi. Tarmoq yo'q bo'lsa oxirgi muvaffaqiyatli javob
rules_cache.json dan o'qiladi — kamera qoidasiz qolmasin.

Login mars-quiz-fix skilli bilan bir xil (oddiy JSON, brauzer kerak emas):
    POST /api/v1/auth/signin, header X-App-Audience: admin

Mustaqil ishga tushirish:
    ./venv/bin/python rules.py           # ro'yxat + kamera qamrovi
"""
import os
import json
import time
import threading

import requests

BASE = "https://api.marsit.uz"
RULES_URL = f"{BASE}/api/v2/reports/audit-rules"
SIGNIN_URL = f"{BASE}/api/v1/auth/signin"

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, "rules_cache.json")
# Parol camera-ai/.env dan yoki mars-quiz-fix skillidan olinadi (nusxa saqlamaymiz).
_ENV_FILES = (
    os.path.join(_HERE, ".env"),
    os.path.expanduser("~/.claude/skills/mars-quiz-fix/.env"),
)

HEADERS = {
    "Accept": "application/json",
    "Origin": "https://core.marsit.uz",
    "Referer": "https://core.marsit.uz/",
    "X-App-Audience": "admin",
}

TTL_SEC = 3600          # ro'yxat shu vaqtdan keyin qayta so'raladi
_cache = {"at": 0.0, "rules": None}
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────── kamera qamrovi
#
# Qaysi qoidani kamera aniqlay oladi. Kalit — qoida raqami (title, masalan "3.1"),
# chunki id baza ko'chirilsa o'zgarishi mumkin, raqam esa hujjatdagi nomer.
#
#   detector : shu qoidani tekshiradigan modul nomi (detectors.py dagi funksiya)
#   needs    : nima kerakligi — kamera o'zi yetarli emas
#   note     : nega shunday qaror qilingani
#
# Bu yerda YO'Q qoida = kamera unga tegmaydi (ovoz, ekran mazmuni, kontekst).
# Ro'yxatni kengaytirish uchun shu jadvalga qator qo'shiladi, qoida matni emas.

COVERAGE = {
    # ── 1-daraja: yuz tanish + zona + dars jadvali yetadi ──────────────
    "3.1": dict(detector="lesson_start", level=1,
                needs=("face", "schedule", "zone"),
                note="Dars boshlanish vaqtida mentor xonada bo'lishi kerak"),
    "3.3": dict(detector="left_room", level=1,
                needs=("face", "schedule", "zone"),
                note="Dars vaqtida mentor xonadan N daqiqadan ko'p yo'qolmasin"),
    "3.5": dict(detector="alone_with_student", level=1,
                needs=("face", "zone"),
                note="Xonada 1 mentor + 1 o'quvchi qolsa — signal"),
    "3.8": dict(detector="admin_zone_loitering", level=1,
                needs=("face", "zone"),
                note="Administratsiya zonasida N daqiqadan ortiq turish"),
    "2.1": dict(detector="late_arrival", level=1,
                needs=("face", "schedule"),
                note="Dars vaqtidan 5 daqiqa oldin ish joyida bo'lish"),
    "2.2": dict(detector="late_arrival", level=1,
                needs=("face", "schedule"),
                note="2.1 bilan bir xil matn — bazada dublikat, bitta detektor"),
    "2.8": dict(detector="lesson_overrun", level=1,
                needs=("face", "schedule", "zone"),
                note="Darsni 5 daqiqadan ko'p kechiktirib tamomlash"),

    # ── 2-daraja: pose/obyekt modeli kerak, aniqlik o'rtacha ───────────
    "2.10": dict(detector="mentor_seated", level=2, needs=("face", "pose", "schedule"),
                 note="Mentor darsni o'tirib o'tmasin"),
    "3.10": dict(detector="sleeping", level=2, needs=("pose", "zone", "schedule"),
                 note="Dars vaqtida uxlash — bosh partada"),
    "3.11": dict(detector="sleeping", level=2, needs=("pose", "zone"),
                 note="Coworkingda uxlash — 3.10 bilan bir detektor, zonasi boshqa"),
    "3.2": dict(detector="phone_in_hand", level=2, needs=("object", "schedule"),
                note="YOLO 'cell phone' klassi; qo'ng'iroqqa 30s ruxsat bor"),
    "3.4": dict(detector="food_drink", level=2, needs=("object",),
                note="YOLO 'bottle'/'cup' — krujkaga ruxsat, rangli ichimlikka yo'q"),
    "2.5": dict(detector="coworking_gathering", level=2, needs=("zone",),
                note="Coworkingda 3+ xodim uzoq turib qolsa"),
    "1.3": dict(detector="chairs_left_untidy", level=2, needs=("zone", "schedule"),
                note="Dars tugagach stullar tartibsizmi — dars oldi/keyin solishtiriladi"),
    "2.3": dict(detector="dress_code", level=2, needs=("face",),
                note="Kiyinish qoidasi — shift kamerasida ishonchsiz, sinash kerak"),

    # ── Bejik: rasmiy qoida, lekin hozirgi usul ishonchsiz ─────────────
    "1.1": dict(detector="badge", level=2, needs=("face",), reliable=False,
                note="Ko'k devor tufayli HSV usuli yolg'on signal beradi. "
                     "Yuz topilgan joydan ko'krak sohasi kesib olinadi"),
    "2.9": dict(detector="badge", level=2, needs=("face",), reliable=False,
                note="1.1 ning eskalatsiyasi: kun ichida 2-marta = ogohlantirish"),
}

# Kamera prinsipial jihatdan tegmaydigan qoidalar — nega ekani yozib qo'yilgan,
# keyingi safar "buni ham qo'shaylik" degan savol qaytmasin.
OUT_OF_SCOPE = {
    "1.2": "Ekran mazmuni — kamera monitorni o'qiy olmaydi",
    "2.4": "1.3 ning dublikati (bazada ikki marta yozilgan)",
    "2.6": "Jihozga ehtiyotkorlik — hodisa emas, baho",
    "2.7": "Muammoni xabar qilish — kameraga ko'rinmaydi",
    "2.11": "Eskalatsiya (2 ta yashil) — bu hisob, kamera emas",
    "3.13": "Eskalatsiya (2 ta sariq) — bu hisob, kamera emas",
    "3.6": "Diniy suhbat — ovoz kerak",
    "3.7": "Muomala sifati — baho, hodisa emas",
    "3.9": "O'yin/kino/ijtimoiy tarmoq — ekran mazmuni",
    "3.12": "Haqorat — ovoz kerak",
    "4.1": "Janjal — texnik mumkin, lekin yolg'on ayblov narxi juda qimmat",
    "4.2": "Chekish — tashqarida (100m radius) kamera yo'q",
    "4.3": "O'quvchini urish — 4.1 bilan bir xil sabab",
    "4.4": "Qizlarga gap otish — ovoz/chat kerak",
}


# ─────────────────────────────────────────────────────────────── kirish
def _creds():
    phone, password = os.environ.get("MARS_PHONE"), os.environ.get("MARS_PASSWORD")
    if phone and password:
        return phone, password
    for path in _ENV_FILES:
        if not os.path.exists(path):
            continue
        vals = {}
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
        if vals.get("MARS_PHONE") and vals.get("MARS_PASSWORD"):
            return vals["MARS_PHONE"], vals["MARS_PASSWORD"]
    raise RuntimeError("MARS_PHONE/MARS_PASSWORD topilmadi")


# Token diskda saqlanadi. Aks holda har ishga tushganda qayta login bo'ladi
# (rules + schedule = 2 ta), va API 429 "Too Many Requests" qaytara boshlaydi —
# ishlab chiqishda shunday bo'ldi. Token muddati tugasa 401 keladi, o'shanda
# yangilanadi.
TOKEN_FILE = os.path.join(_HERE, ".token")
_token_cache = [None]


def _login() -> str:
    phone, password = _creds()
    r = requests.post(SIGNIN_URL, json={"user": {"phone": phone, "password": password}},
                      headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError(f"login rad etildi: {r.text[:200]}")
    token = data["access_token"]
    _token_cache[0] = token
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({"access_token": token}, f)
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass      # token saqlanmasa ham ishlayveradi, faqat qayta login bo'ladi
    return token


def _token(refresh=False) -> str:
    if refresh:
        return _login()
    if _token_cache[0]:
        return _token_cache[0]
    if os.path.exists(TOKEN_FILE):
        try:
            token = json.load(open(TOKEN_FILE))["access_token"]
            _token_cache[0] = token
            return token
        except Exception:
            pass
    return _login()


def authed_get(url, **kwargs):
    """Tokenli GET. 401 kelsa bir marta qayta login qilib urinadi."""
    for refresh in (False, True):
        h = dict(HEADERS)
        h["Authorization"] = f"Bearer {_token(refresh=refresh)}"
        r = requests.get(url, headers=h, timeout=30, **kwargs)
        if r.status_code != 401:
            r.raise_for_status()
            return r
    r.raise_for_status()
    return r


def _fetch():
    r = authed_get(RULES_URL)
    rules = r.json()
    if not isinstance(rules, list) or not rules:
        raise RuntimeError(f"kutilmagan javob: {str(rules)[:200]}")
    return rules


def load(force=False):
    """Qoidalar ro'yxati. Avval API, u yiqilsa keshdan.

    Qaytaradi: [{id, title, description, score, type, category}, ...]
    """
    with _lock:
        fresh = _cache["rules"] is not None and time.time() - _cache["at"] < TTL_SEC
        if fresh and not force:
            return _cache["rules"]
        try:
            rules = _fetch()
            _cache.update(at=time.time(), rules=rules)
            with open(CACHE_FILE, "w") as f:
                json.dump({"at": time.time(), "rules": rules}, f,
                          ensure_ascii=False, indent=1)
            return rules
        except Exception as e:
            if _cache["rules"] is not None:
                print(f"[rules] API javob bermadi ({e}) — xotiradagi ro'yxat ishlatiladi")
                return _cache["rules"]
            if os.path.exists(CACHE_FILE):
                cached = json.load(open(CACHE_FILE))
                age_h = (time.time() - cached.get("at", 0)) / 3600
                print(f"[rules] API javob bermadi ({e}) — kesh ishlatiladi "
                      f"({age_h:.0f} soat oldingi)")
                _cache.update(at=time.time(), rules=cached["rules"])
                return cached["rules"]
            raise


def by_number(number: str):
    """Qoidani raqami bo'yicha topadi ("3.1"). Topilmasa None.

    Bazada bir raqam ikki marta uchraydi (1.3) — tavsifi bo'lganini qaytaramiz,
    chunki bo'sh qoidani jarima sifatida qo'yib bo'lmaydi.
    """
    found = [r for r in load() if (r.get("title") or "").strip() == number]
    if not found:
        return None
    with_text = [r for r in found if (r.get("description") or "").strip()]
    return (with_text or found)[0]


def camera_rules(level=None, include_unreliable=True):
    """Kamera tekshiradigan qoidalar: [(qoida, qamrov), ...].

    level=1 — faqat ishonchli (yuz + zona + jadval)
    level=2 — pose/obyekt modeliga tayanadigan qo'shimchalar
    """
    out = []
    for number, cov in COVERAGE.items():
        if level is not None and cov["level"] != level:
            continue
        if not include_unreliable and cov.get("reliable") is False:
            continue
        rule = by_number(number)
        if rule is None:
            print(f"[rules] DIQQAT: {number} qoidasi bazada yo'q — "
                  f"o'chirilgan yoki raqami o'zgargan, detektor o'chiriladi")
            continue
        out.append((rule, cov))
    return out


def detectors_needed(level=None):
    """Qaysi detektorlar kerak: {detector_nomi: [qoida raqamlari]}."""
    need = {}
    for rule, cov in camera_rules(level=level):
        need.setdefault(cov["detector"], []).append(rule["title"])
    return need


def audit(rules=None):
    """Baza va COVERAGE mos kelyaptimi — nomuvofiqliklar ro'yxati."""
    rules = rules or load()
    numbers = {(r.get("title") or "").strip() for r in rules}
    problems = []
    for number in COVERAGE:
        if number not in numbers:
            problems.append(f"COVERAGE da {number} bor, bazada yo'q")
    for number in OUT_OF_SCOPE:
        if number not in numbers:
            problems.append(f"OUT_OF_SCOPE da {number} bor, bazada yo'q")
    for r in rules:
        number = (r.get("title") or "").strip()
        if number not in COVERAGE and number not in OUT_OF_SCOPE:
            problems.append(f"{number} (id={r['id']}) hech qayerda tasniflanmagan")
        if not (r.get("description") or "").strip():
            problems.append(f"{number} (id={r['id']}) tavsifi bo'sh — jarima qo'yib bo'lmaydi")
    return problems


if __name__ == "__main__":
    rules = load(force=True)
    print(f"Bazadan olindi: {len(rules)} ta faol qoida\n")

    for lvl in (1, 2):
        pairs = camera_rules(level=lvl)
        print(f"── {lvl}-daraja: {len(pairs)} ta qoida")
        for rule, cov in sorted(pairs, key=lambda p: p[0]["title"]):
            mark = "" if cov.get("reliable") is not False else "  [ishonchsiz]"
            print(f"   {rule['title']:<5} {rule['type']:<6} {rule['score']:>4} ball  "
                  f"→ {cov['detector']}{mark}")
        print()

    print(f"── Kamera tegmaydi: {len(OUT_OF_SCOPE)} ta")
    for number, why in sorted(OUT_OF_SCOPE.items()):
        print(f"   {number:<5} {why}")
    print()

    print("── Kerak bo'ladigan detektorlar")
    for name, numbers in sorted(detectors_needed().items()):
        print(f"   {name:<22} {', '.join(sorted(numbers))}")
    print()

    problems = audit(rules)
    print(f"── Nomuvofiqliklar: {len(problems)}")
    for p in problems:
        print(f"   • {p}")
