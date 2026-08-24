"""Davomat — xodim qachon keldi, qachon ketdi.

Manba: yuz tanish. Kamera tanigan har bir xodim uchun kun davomida:
    keldi  = shu kuni BIRINCHI marta ko'ringan vaqt
    ketdi  = shu kuni OXIRGI marta ko'ringan vaqt (aniqrog'i — oxirgi
             ko'rinish; odam eshikdan chiqqanini kamera bilmaydi)

Halol cheklov: tanish faqat yuz katta ko'rinadigan kameralarda ishlaydi
(kirish/coworking; dars xonalarida yuz 15-40 piksel — tanilmaydi). Ya'ni
davomat "binoga kirdi-chiqdi" emas, "tanish ishlaydigan kameralar oldidan
o'tdi" degani. Kirish eshigiga qaragan kamera bo'lsa, aynan keldi-ketti
bo'ladi.

Saqlash: data/attendance/YYYY-MM-DD.json
    {ism: {first, last, first_cam, last_cam, seen}}
Yozish atomik (tmp + rename) va faqat o'zgarish bo'lganda.
"""
import json
import os
import threading
import time
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(_HERE, "data", "attendance")
os.makedirs(DIR, exist_ok=True)

# Bir ko'rinishdan keyin shu vaqt ichida qayta yozib o'tirmaymiz — analizator
# sekundiga bir necha marta tanishi mumkin, diskka har safar yozish shart emas.
MIN_UPDATE_SEC = 30

_lock = threading.Lock()
_cache = {"date": None, "data": {}}
_last_write = {}


def _path(day):
    return os.path.join(DIR, f"{day}.json")


def _load(day):
    p = _path(day)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(day, data):
    tmp = _path(day) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(day))


def record(name, branch, camera, when=None):
    """Xodim ko'rindi. Birinchi ko'rinish = keldi, har keyingisi last ni suradi."""
    now = when or time.time()
    day = datetime.date.fromtimestamp(now).isoformat()
    hhmm = datetime.datetime.fromtimestamp(now).strftime("%H:%M:%S")
    where = f"{branch}/{camera}"

    with _lock:
        if _cache["date"] != day:
            _cache.update(date=day, data=_load(day))
            _last_write.clear()

        data = _cache["data"]
        entry = data.get(name)
        changed = False
        if entry is None:
            data[name] = {"first": hhmm, "last": hhmm,
                          "first_cam": where, "last_cam": where, "seen": 1}
            changed = True
        else:
            entry["seen"] += 1
            # last ni har doim yangilaymiz, lekin diskka MIN_UPDATE_SEC da bir
            entry["last"] = hhmm
            entry["last_cam"] = where
            if now - _last_write.get(name, 0) >= MIN_UPDATE_SEC:
                changed = True
        if changed:
            _last_write[name] = now
            _save(day, data)


def day(date_str=None):
    """Kun davomati: {ism: {first, last, first_cam, last_cam, seen}}."""
    d = date_str or datetime.date.today().isoformat()
    with _lock:
        if _cache["date"] == d:
            # nusxa — chaqiruvchi o'zgartirib yubormasin
            return d, {k: dict(v) for k, v in _cache["data"].items()}
    return d, _load(d)


def days():
    """Yozuvi bor kunlar ro'yxati (yangi birinchi)."""
    out = []
    for f in os.listdir(DIR):
        if f.endswith(".json") and not f.endswith(".tmp"):
            out.append(f[:-5])
    return sorted(out, reverse=True)


if __name__ == "__main__":
    d, data = day()
    print(f"Davomat {d}: {len(data)} xodim")
    for name, e in sorted(data.items(), key=lambda kv: kv[1]["first"]):
        print(f"  {name:<25} keldi {e['first']}  oxirgi {e['last']}  "
              f"({e['seen']} ko'rinish, {e['last_cam']})")
