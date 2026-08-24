"""Detektor mantiqi sinovi — kamera va tarmoqsiz, soxta kadr bilan.

Muhim savol: detektor bir kadrdan signal bermaydimi (yolg'on jarima) va
chegaradan oshganda haqiqatan beradimi.

    ./venv/bin/python test_detectors.py
"""
import datetime

import detectors
import schedule


def ctx(zone_channel, faces=None, persons=None, now=None, branch="Yunusobod",
        camera_name="B3"):
    return detectors.Context(branch=branch, channel=zone_channel,
                             camera_name=camera_name, faces=faces or [],
                             persons=persons or [], now=now)


def face(name):
    return {"box": (0, 0, 50, 50), "name": name, "score": 0.9}


def person(seated=False, head_down=False, reliable=True):
    return {"box": (0, 0, 50, 150), "height": 150, "conf": 0.9,
            "keypoints": [], "reliable": reliable,
            "seated": seated, "head_down": head_down}


def check(label, ok):
    print(f"  {'OK  ' if ok else 'XATO'} {label}")
    return ok


def main():
    passed = failed = 0

    def run(label, ok):
        nonlocal passed, failed
        if check(label, ok):
            passed += 1
        else:
            failed += 1

    print("── Zona xaritasi")
    run("B3 (1601) dars xonasi",
        ctx("1601").zone == detectors.CLASSROOM)
    run("Admin-kassa (101) admin zonasi",
        ctx("101", camera_name="Admin-kassa").zone == detectors.ADMIN)
    run("Coworking (1201) coworking",
        ctx("1201", camera_name="Coworking").zone == detectors.COWORKING)

    print("\n── Bir kadrdan signal bermaydi (yolg'on jarimaga qarshi)")
    det = detectors.AdminZoneLoitering()
    c = ctx("101", faces=[face("Akrom")], camera_name="Admin-kassa")
    run("admin zonasida 1-kadr — signal yo'q", det.check(c) == [])

    print("\n── Chegaradan oshsa signal beradi")
    det = detectors.AdminZoneLoitering()
    t0 = datetime.datetime.now()
    # 6 daqiqa uzluksiz turgan holatni taqlid qilamiz
    events = []
    for minutes in (0, 2, 4, 6):
        c = ctx("101", faces=[face("Akrom")], camera_name="Admin-kassa",
                now=t0 + datetime.timedelta(minutes=minutes))
        events += det.check(c)
    run("admin zonasida 6 daqiqadan keyin — signal bor", len(events) == 1)
    if events:
        e = events[0]
        run("signal 3.8 qoidasiga tegishli", e["rule_number"] == "3.8")
        run("qoida matni API'dan keldi", bool(e["rule_text"]))
        run("ball manfiy", e["score"] < 0)
        print(f"       → {e['reason']}")

    print("\n── Sovish vaqti (bitta hodisa takror signal bermaydi)")
    c = ctx("101", faces=[face("Akrom")], camera_name="Admin-kassa",
            now=t0 + datetime.timedelta(minutes=8))
    run("darhol qayta signal bermaydi", det.check(c) == [])

    print("\n── Ishonchsiz kadr signal bermaydi")
    det = detectors.Sleeping()
    t0 = datetime.datetime.now()
    for minutes in (0, 1, 2):
        c = ctx("1201", camera_name="Coworking",
                persons=[person(head_down=True, reliable=False)],
                now=t0 + datetime.timedelta(minutes=minutes))
        ev = det.check(c)
    run("pose ishonchsiz bo'lsa — signal yo'q", ev == [])

    print("\n── Ishonchli uxlash signali (coworking, 3.11)")
    det = detectors.Sleeping()
    fired = []
    for minutes in (0, 1, 2):
        c = ctx("1201", camera_name="Coworking",
                persons=[person(head_down=True)],
                now=t0 + datetime.timedelta(minutes=minutes))
        fired += det.check(c)
    run("coworkingda uxlash — 3.11 signali",
        len(fired) == 1 and fired[0]["rule_number"] == "3.11")
    run("sovish vaqti takror signalni to'sdi", len(fired) == 1)
    if fired:
        print(f"       → {fired[0]['reason']}")

    print("\n── Dars xonasida uxlash faqat dars vaqtida (3.10)")
    det = detectors.Sleeping()
    # Dars yo'q vaqt — yarim tunda hech qanday dars bo'lmaydi
    midnight = datetime.datetime.combine(datetime.date.today(),
                                         datetime.time(3, 0))
    fired = []
    for minutes in (0, 1, 2):
        c = ctx("1601", camera_name="B3", persons=[person(head_down=True)],
                now=midnight + datetime.timedelta(minutes=minutes))
        fired += det.check(c)
    run("dars yo'q vaqtda xonada uxlash — signal yo'q", fired == [])

    print("\n── Streak: uzilish holatni bekor qiladi")
    s = detectors.Streak(gap_tol=10.0)
    s.update("k", True, 100.0)
    run("30 sekund uzilishdan keyin nolga tushadi",
        s.update("k", False, 130.0) == 0.0 and s.update("k", True, 131.0) < 1.0)

    print(f"\n{'='*46}\n{passed} ta o'tdi, {failed} ta yiqildi")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
