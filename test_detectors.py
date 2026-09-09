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


def _page_js_ok():
    """Brauzerga KETAYOTGAN sahifa skripti ishga tushadimi.

    Ikki marta shu yerda tutilmagan xato o'tib ketdi, chunki sinov app.py
    MATNIDAN skript ajratardi — Python qatorining o'zidan, ya'ni Python
    escape'lari yechilmagan holidan. Brauzerga esa YECHILGAN qiymat boradi
    va farq aynan apostrofda chiqadi: "yo'q" satrni uzib yuboradi.
    Shuning uchun endi app.PAGE QIYMATI tekshiriladi.

    Bundan tashqari sintaksis yetarli emas: skript ishga tushganda ham
    yiqilishi mumkin. Shuning uchun u soxta DOM'da HAQIQATAN bajariladi.
    """
    import re
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        print("       (node yo'q — o'tkazib yuborildi)")
        return True
    import app as _app
    m = re.search(r"<script>(.*?)</script>", _app.PAGE, re.S)
    if not m:
        print("       skript topilmadi")
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(m.group(1))
        path = f.name
    r = subprocess.run([node, "tools_domcheck.js", path],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode:
        print("      ", out.split("\n")[0])
    return r.returncode == 0


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

    print("\n── Yuz o'qilmasa ayblov qo'yilmaydi (jonli sinovda topilgan)")
    # B3 holati: dars ketyapti, 6 kishi bor, yuzlar 18-41px — hech kim tanilmadi.
    # Bu "mentor yo'q" degani EMAS, "ko'ra olmadim" degani.
    small = [{"box": (0, 0, 25, 30), "name": None, "score": 0.25}
             for _ in range(3)]
    c = ctx("1601", faces=small, persons=[person() for _ in range(6)])
    run("kichik yuzlar — identity ishonchsiz", c.identity_reliable is False)
    det = detectors.LeftRoom()
    fired = []
    for minutes in (0, 3, 6, 9):
        c = ctx("1601", faces=small, persons=[person() for _ in range(6)],
                now=datetime.datetime.combine(datetime.date.today(),
                                              datetime.time(10, 20))
                    + datetime.timedelta(minutes=minutes))
        fired += det.check(c)
    run("dars vaqtida ham yolg'on 'mentor yo'q' signali yo'q", fired == [])

    big = [{"box": (0, 0, 90, 110), "name": None, "score": 0.3}]
    c = ctx("1601", faces=big)
    run("katta yuz — identity ishonchli", c.identity_reliable is True)
    c = ctx("1601", faces=[{"box": (0, 0, 20, 25), "name": "Akrom", "score": 0.7}])
    run("kichik bo'lsa ham tanilgan bo'lsa — ishonchli",
        c.identity_reliable is True)

    print("\n── Streak: uzilish holatni bekor qiladi")
    s = detectors.Streak(gap_tol=10.0)
    s.update("k", True, 100.0)
    run("30 sekund uzilishdan keyin nolga tushadi",
        s.update("k", False, 130.0) == 0.0 and s.update("k", True, 131.0) < 1.0)

    print("\n── Odam sanash")
    # head_count "odam bormi" ni sanashi kerak, "holati o'qiladimi" ni emas.
    # Bu farq jonli sinovda chiqdi: stolga yarim yashiringan odam
    # reliable=False bo'lgani uchun umuman sanalmasdi.
    hidden = [person(reliable=False) for _ in range(3)]
    c = ctx("1601", persons=hidden)
    run("holati o'qilmaydigan odam ham sanaladi", c.head_count == 3)
    c = ctx("1601", persons=[person(), person(reliable=False)])
    run("aralash holatda hammasi sanaladi", c.head_count == 2)
    c = ctx("1601", faces=[{"box": (0, 0, 90, 110), "name": None, "score": 0.3}])
    run("pose bo'lmasa yuz bo'yicha sanaydi", c.head_count == 1)

    print("\n── Dublikat qutilarni yig'ish")
    import pose as _pose
    # Bir odamga ikkita quti: biri ikkinchisining ichida (jonli holat, B1)
    big_box, small_box = (100, 100, 200, 400), (120, 120, 190, 280)
    run("ichma-ich quti dublikat deb topiladi",
        _pose._inside(small_box, big_box) >= _pose.DUP_INSIDE)
    run("uzoq qutilar dublikat emas",
        _pose._inside((100, 100, 200, 400), (500, 500, 600, 800)) == 0.0)

    print("\n── Xodim bazasi")
    import faces as _f
    n0 = len(_f.people())
    run("baza o'qildi", n0 > 0)
    run("har xodimda namuna bor", all(p["samples"] >= 1 for p in _f.people()))
    ok, msg = _f.enroll(None, "")          # ism yo'q — kadrga ham tegmasligi kerak
    run("ismsiz qo'shishni rad etadi", ok is False and "Ism" in msg)
    ok, msg = _f.remove("__yo'q__")
    run("yo'q xodimni o'chirishni rad etadi", ok is False)
    run("baza o'zgarmadi", len(_f.people()) == n0)
    # Yuz o'lchami chegaralari mantiqan to'g'rimi
    run("ro'yxatga olish chegarasi tanishnikidan qattiq",
        _f.MIN_ENROLL_PX > _f.MIN_RECOGNIZE_PX)

    print("\n── 3.2 Telefon qo'lda")
    import pose as _pz
    # Bog'lash: telefon bilak yonida → qo'lda; stolda (odamdan tashqarida) → yo'q
    kp = [[0, 0, 0.0]] * 17
    kp[_pz.L_WR] = [40, 90, 0.9]           # chap bilak
    holder = {"box": (0, 0, 80, 200), "height": 200, "keypoints": kp,
              "reliable": True, "seated": None, "head_down": None}
    other = {"box": (300, 0, 380, 200), "height": 200, "keypoints": None,
             "reliable": False, "seated": None, "head_down": None}
    _pz._attach_phones([holder, other], [(35, 85, 50, 100, 0.6),   # bilakda
                                          (600, 300, 620, 320, 0.7)])  # stolda
    run("bilak yonidagi telefon odamga bog'lanadi", holder["phone"] is not None)
    run("stoldagi telefon hech kimga bog'lanmaydi", other["phone"] is None)
    far = dict(holder, phone=None)
    _pz._attach_phones([far], [(40, 190, 55, 200, 0.6)])        # bilakdan uzoq (oyoqda)
    run("bilakdan uzoq telefon qo'lda emas", far["phone"] is None)
    nokp = dict(other, phone=None)
    _pz._attach_phones([nokp], [(330, 40, 345, 60, 0.6)])       # bo'g'imsiz, yuqori qism
    run("bo'g'imsiz odamda yuqoridagi telefon bog'lanadi", nokp["phone"] is not None)

    # Vaqt: coworking sinov rejimida (PHONE_ZONES) bir kadr signal emas, 45s dan keyin bor
    _saved = detectors.PHONE_ZONES
    detectors.PHONE_ZONES = {detectors.CLASSROOM, detectors.COWORKING}
    try:
        det = detectors.PhoneInHand()
        t0 = datetime.datetime.now()
        ph = {"box": (10, 10, 20, 20), "conf": 0.6, "wrist": 0.1}
        fired = []
        for sec in (0, 10, 20, 30, 40):
            c = ctx("201", camera_name="Coworking 1",
                    persons=[dict(person(), tid=7, phone=ph)],
                    now=t0 + datetime.timedelta(seconds=sec))
            fired += det.check(c)
        run("45s gacha hodisa yo'q", fired == [])
        c = ctx("201", camera_name="Coworking 1",
                persons=[dict(person(), tid=7, phone=ph)],
                now=t0 + datetime.timedelta(seconds=50))
        fired += det.check(c)
        run("45s dan keyin 3.2 hodisa", len(fired) == 1 and fired[0]["rule_number"] == "3.2")
        # Telefonsiz odam — hech qachon
        det2 = detectors.PhoneInHand()
        r = []
        for sec in (0, 30, 60, 90):
            r += det2.check(ctx("201", camera_name="Coworking 1",
                                persons=[dict(person(), tid=1, phone=None)],
                                now=t0 + datetime.timedelta(seconds=sec)))
        run("telefonsiz — hodisa yo'q", r == [])
    finally:
        detectors.PHONE_ZONES = _saved
    # Standart rejim: coworking zonasi tegmaydi
    det3 = detectors.PhoneInHand()
    r3 = []
    for sec in (0, 60, 120):
        r3 += det3.check(ctx("201", camera_name="Coworking 1",
                             persons=[dict(person(), tid=7, phone=ph)],
                             now=t0 + datetime.timedelta(seconds=sec)))
    run("standart rejimda coworking tegmaydi", r3 == [])

    print("\n── Sig'im qoidasi (mahalliy)")
    det=detectors.Capacity()
    t0=datetime.datetime.now()
    # 5 kishi (chegara 4) — 5 sekunddan keyin hodisa
    fired=[]
    for sec in (0,2,4,6):
        c=ctx("1101", persons=[person() for _ in range(5)], camera_name="B4",
              now=t0+datetime.timedelta(seconds=sec))
        fired+=det.check(c)
    run("4 dan ko'p bo'lsa hodisa chiqadi", len(fired)==1)
    if fired:
        run("mahalliy hodisa (ball yo'q)", fired[0]["rule_number"]=="LOKAL")
        run("odam soni xabarda", "5 kishi" in fired[0]["reason"])
    # 4 kishi (chegarada) — hodisa yo'q
    det2=detectors.Capacity()
    for sec in (0,3,6):
        c=ctx("1101", persons=[person() for _ in range(4)], camera_name="B4",
              now=t0+datetime.timedelta(seconds=sec))
        r2=det2.check(c)
    run("chegarada (4) hodisa yo'q", r2==[])
    # sig'im qo'yilmagan kamera — tegmaydi
    c=ctx("1601", persons=[person() for _ in range(9)], camera_name="B3")
    run("sig'im yo'q kamera — hodisa yo'q", det2.check(c)==[])

    print("\n── Davomat")
    import os, time as _t, shutil as _sh, datetime as _dt
    import attendance as _att
    # ALOHIDA (o'tmish) sanaga yozamiz — jonli ilova bugungi faylni yozsa ham
    # to'qnashmasin. 09:00 + 3600 = 10:00, SHU kunда qoladi (yarim tundan o'tmaydi).
    _fix="1990-01-01"; _p=_att._path(_fix); _bak=None
    if os.path.exists(_p): _bak=_p+".bak"; _sh.copy(_p,_bak)
    _att._cache.update(date=None, data={}); _att._last_write.clear()
    try:
        t0=_dt.datetime.combine(_dt.date(1990,1,1), _dt.time(9,0)).timestamp()
        _att.record("__SINOV__","F","K1",when=t0)
        _att.record("__SINOV__","F","K1",when=t0+5)
        _att.record("__SINOV__","F","K2",when=t0+3600)
        _,_data=_att.day(_fix)
        e=_data.get("__SINOV__")
        run("birinchi ko'rinish = keldi", e is not None and e["first_cam"]=="F/K1")
        run("oxirgi ko'rinish = ketdi", e["last_cam"]=="F/K2" and e["first"]!=e["last"])
        run("ko'rinishlar sanaladi", e["seen"]==3)
    finally:
        if _bak: os.replace(_bak,_p)
        elif os.path.exists(_p): os.remove(_p)
        _att._cache.update(date=None, data={})

    print("\n── Dashboard skripti")
    run("brauzerdagi skript ishga tushadi", _page_js_ok())

    print(f"\n{'='*46}\n{passed} ta o'tdi, {failed} ta yiqildi")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
