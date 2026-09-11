"""Mehmon xotirasi sinovi — kamera va Telegramsiz, soxta vektorlar bilan.

Asosiy savol: bir odam ketib qaytsa, IKKINCHI xabar chiqmaydimi; boshqa
odam kelsa, yangi mehmon bo'ladimi; TTL o'tgach yana yangi bo'ladimi.

    ./venv/bin/python test_visitors.py
"""
import os
import tempfile
import time

import numpy as np

import visitors

rng = np.random.default_rng(7)


def unit():
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def near(e, noise=0.25):
    v = e + rng.normal(size=512).astype(np.float32) * noise / 512 ** 0.5
    return v / np.linalg.norm(v)


def fresh():
    path = os.path.join(tempfile.mkdtemp(), "visitors.json")
    return visitors.Memory(path)


def test_same_person_returns_no_new_visitor():
    m = fresh()
    a = unit()
    v = m.add(a, "B4", now=1000)
    assert v["n"] == 1
    # 5 daqiqadan keyin qaytdi — biroz boshqa burchak
    got, how, score = m.match(near(a), now=1300)
    assert got is not None and got["id"] == v["id"] and how == "yuz", score
    assert m.seen(got, near(a), "B4", now=1300) is True     # qaytish
    assert m.seen(got, near(a), "B4", now=1301) is False    # o'sha uzluksiz
    assert len(m.active(now=1301)) == 1
    assert len(got["seen"]) == 2


def test_other_person_is_new():
    m = fresh()
    m.add(unit(), "B4", now=1000)
    got, how, score = m.match(unit(), now=1001)
    assert got is None, score
    v2 = m.add(unit(), "B4", now=1001)
    assert v2["n"] == 2


def test_ttl_expires():
    m = fresh()
    a = unit()
    m.add(a, "B4", now=1000)
    got, _, _ = m.match(a, now=1000 + visitors.TTL_SEC + 1)
    assert got is None
    assert m.active(now=1000 + visitors.TTL_SEC + 1) == []


def test_persists_across_restart():
    m = fresh()
    a = unit()
    now = time.time()                  # yuklashda haqiqiy vaqt bilan TTL tekshiriladi
    v = m.add(a, "B4", now=now)
    m2 = visitors.Memory(m.path)
    got, _, _ = m2.match(a, now=now)
    assert got is not None and got["id"] == v["id"]


def test_embs_capped_and_diverse():
    m = fresh()
    a = unit()
    v = m.add(a, "B4", now=1000)
    m.seen(v, a, now=1000)                 # bir xil — saqlanmaydi
    assert len(v["embs"]) == 1
    for _ in range(10):
        m.seen(v, near(a, 0.6), now=1000)
    assert len(v["embs"]) <= visitors.MAX_EMBS


def test_caption_lists_returns():
    v = {"n": 3, "camera": "B4", "first": 0, "seen": [0, 400, 800]}
    c = visitors.caption(v)
    assert "Begona mijoz #3" in c and "Yana ko'rindi" in c


def main():
    tests = [(k, f) for k, f in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"{len(tests)} sinov o'tdi")


def test_good_face_gate():
    ok = {"box": (0, 0, 120, 120), "sharp": 40, "bright": 120, "pose": [5, -10, 0]}
    assert visitors.good_face(ok) == (True, "")
    assert visitors.good_face(dict(ok, bright=25))[1] == "qorong'i"
    assert visitors.good_face(dict(ok, sharp=1))[1] == "xira"
    assert visitors.good_face(dict(ok, pose=[10, 60, 0]))[1] == "burilgan"
    assert visitors.good_face(dict(ok, pose=[50, 0, 0]))[1] == "burilgan"
    assert visitors.good_face(dict(ok, box=(0, 0, 30, 30)))[1] == "kichik"


def test_body_matches_without_face():
    m = fresh()
    b = unit()
    v = m.add(None, "Mac", now=1000, body_emb=b)
    got, how, sc = m.match(None, near(b, 0.4), now=1200)
    assert got is not None and got["id"] == v["id"] and how == "tana", (how, sc)


def test_same_clothes_different_face_is_new():
    """Bir xil kiyim, lekin yuzlar aniq farq — BOSHQA odam."""
    m = fresh()
    b, f1, f2 = unit(), unit(), unit()
    m.add(f1, "Mac", now=1000, body_emb=b)
    got, how, sc = m.match(f2, near(b, 0.2), now=1100)
    assert got is None, (how, sc)


def test_same_clothes_no_face_but_excluded_on_screen():
    """Ikkinchi odam ekranda birinchi bilan bir vaqtda — nomzod emas."""
    m = fresh()
    b = unit()
    v = m.add(None, "Mac", now=1000, body_emb=b)
    got, how, sc = m.match(None, near(b, 0.2), now=1001, exclude={v["id"]})
    assert got is None


def test_body_geometry_conflict_is_new():
    m = fresh()
    b = unit()
    m.add(None, "Mac", now=1000, body_emb=b, geom={"shoulder_ratio": 0.20})
    got, how, sc = m.match(None, near(b, 0.2), geom={"shoulder_ratio": 0.40}, now=1100)
    assert got is None


def test_face_wins_over_body():
    m = fresh()
    b, f = unit(), unit()
    v1 = m.add(f, "Mac", now=1000, body_emb=unit())
    m.add(None, "Mac", now=1000, body_emb=b)
    got, how, sc = m.match(near(f), near(b, 0.2), now=1100)
    assert got["id"] == v1["id"] and how == "yuz"


def test_daily_numbering_restarts():
    m = fresh()
    d1 = time.mktime(time.strptime("2026-09-10 10:00", "%Y-%m-%d %H:%M"))
    d2 = d1 + 86400
    assert m.add(unit(), "Mac", now=d1)["n"] == 1
    assert m.add(unit(), "Mac", now=d1 + 60)["n"] == 2
    assert m.add(unit(), "Mac", now=d2)["n"] == 1          # ertaga yana 1 dan
    assert m.add(unit(), "Mac", now=d2 + 60)["n"] == 2


def test_crowd_and_face_mapping_helpers():
    import identity
    a = {"box": (0, 0, 100, 300), "tid": 1}
    b = {"box": (60, 0, 160, 300), "tid": 2}      # a bilan 40% kesishadi
    c = {"box": (500, 0, 600, 300), "tid": 3}
    assert identity._crowded([a, b, c]) == {0, 1}
    assert identity._overlapping([a, b, c]) == set()   # IoU 0.25 < 0.55 — dublikat emas


def test_crowd_body_threshold_stricter():
    m = fresh()
    b = unit()
    m.add(None, "Mac", now=1000, body_emb=b)
    q = near(b, 0.7)
    got_n, how_n, sc = m.match(None, q, now=1100)
    got_c, how_c, sc_c = m.match(None, q, now=1100, body_thr=0.75)
    # oddiy chegarada mos, olomon chegarasida (qattiq) mos emas bo'lishi mumkin — kamida qattiqroq
    assert (got_c is None) or (sc_c >= 0.75)


def test_identity_unbinds_on_face_contradiction():
    """Trek A mehmonga bog'langan; keyin shu trekda BOSHQA yuz ko'rindi → uziladi."""
    import identity, numpy as np
    m = fresh(); visitors.MEMORY = m
    visitors.STRANGER_CAMS.add("T")
    ci = identity.CamIdentity("T")
    fa, fb = unit(), unit()
    v = m.add(fa, "T", now=time.time())
    ci.names[1] = {"name": None, "visitor": v["id"], "n": v["n"], "box": (0, 0, 50, 50), "score": 0.9, "how": "yuz"}
    p = {"box": (0, 0, 100, 300), "tid": 1, "keypoints": None}
    face = {"box": (20, 10, 80, 80), "name": None, "score": 0.05, "emb": fb, "sharp": 50, "bright": 120, "pose": [0, 0, 0], "too_small": False}
    frame = np.zeros((400, 400, 3), np.uint8)
    import body
    body.embed = lambda fr, ps: [None] * len(ps)      # tana modeli yuklanmasin
    ci.step(frame, [p], [face], True)
    assert 1 not in ci.names or ci.names[1].get("visitor") != v["id"]


if __name__ == "__main__":
    main()
