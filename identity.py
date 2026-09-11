"""Kim bu? — har trek uchun qaror: XODIM / MEHMON / hali noma'lum.

Ikki belgi bilan ishlaydi:
  yuz vektori (ArcFace)  — kuchli, doimiy, lekin yuz ko'rinishi shart
  tana vektori (ReID)    — orqasi bilan ham ishlaydi, lekin faqat o'sha kun
                           va bir xil kiyimda adashishi mumkin

Shuning uchun uch qat'iy qoida (Sardor talabi: bir xil kiyimdagi ikki odam
HECH QACHON bitta mehmon bo'lmasin):
  1. Bir vaqtda ekranda turgan ikki trek — ikki xil odam. Boshqa trekka
     bog'langan mehmon/xodim bu trek uchun nomzod EMAS.
  2. Yuz bor bo'lsa yuz hal qiladi. Tana o'xshasa-yu, yuzlar aniq farq qilsa
     (< FACE_CONTRA) — boshqa odam.
  3. Tana mosligi yolg'iz yetmaydi: chegara yuqori (BODY_THRESHOLD) va
     qomat (yelka/bo'y nisbati) qarama-qarshi bo'lmasin.

Trek "hit" yig'adi (har FACE_INTERVAL da bir): {face|None, body|None, geom}.
STRANGER_MIN_HITS ta hit → qaror. Xodim yuzi tanilsa — darhol xodim.
Xodimning tana vektori ham kun davomida eslab qolinadi (staff_body): xodim
keyin orqasi bilan tursa ham mehmon bo'lib ketmaydi.
"""
import time

import numpy as np

import body
import visitors

STAFF_BODY_MAX = 8


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1, ua)


def _crowded(persons, thr=None):
    """Qutisi boshqa quti bilan (o'z maydoniga nisbatan) ko'p kesishgan odamlar."""
    thr = thr if thr is not None else visitors.CROWD_OVERLAP
    out = set()
    for i, a in enumerate(persons):
        ax1, ay1, ax2, ay2 = a["box"]
        area = max(1, (ax2 - ax1) * (ay2 - ay1))
        for j, b in enumerate(persons):
            if i == j:
                continue
            ix = max(0, min(ax2, b["box"][2]) - max(ax1, b["box"][0]))
            iy = max(0, min(ay2, b["box"][3]) - max(ay1, b["box"][1]))
            if ix * iy / area >= thr:
                out.add(i)
                break
    return out


def _overlapping(persons, thr=0.55):
    """Bir odamga ikki trek (ByteTrack dublikati) — kichigining indekslari."""
    out = set()
    for i, a in enumerate(persons):
        for j, b in enumerate(persons):
            if i >= j:
                continue
            if _iou(a["box"], b["box"]) >= thr:
                area = lambda q: (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1])
                out.add(i if area(a) < area(b) else j)
    return out


def _head_box(p):
    """Yuz ko'rinmaganda label uchun bosh joyi (x, y, w, h) — quti tepasi."""
    x1, y1, x2, y2 = [int(t) for t in p["box"]]
    w, h = x2 - x1, y2 - y1
    s = max(20, int(min(w, h * 0.25)))
    return (x1 + (w - s) // 2, y1, s, s)


class CamIdentity:
    """Bitta kamera uchun trek → shaxs holati."""

    def __init__(self, cam_name):
        self.cam = cam_name
        self.names = {}        # {tid: {name|None, visitor|None, n, box, score, how}}
        self.hits = {}         # {tid: [ {face, body, geom, staff_score} ]}
        self.staff_body = {}   # {ism: [body emb...]} — bugun (kun almashsa tozalanadi)
        self._staff_day = time.strftime("%Y-%m-%d")
        self._diag_last = {}

    # ── yordamchi ────────────────────────────────────────────────────
    def _diag(self, tid, msg):
        key = (tid, msg.split(" ")[0])
        if time.time() - self._diag_last.get(key, 0) >= 5:
            self._diag_last[key] = time.time()
            print(f"[kim?] {self.cam} trek {tid}: {msg}")

    def _gc(self, active):
        for d in (self.names, self.hits):
            for tid in list(d):
                if tid not in active:
                    del d[tid]
        day = time.strftime("%Y-%m-%d")
        if day != self._staff_day:
            self.staff_body.clear()
            self._staff_day = day

    def _remember_staff_body(self, name, emb):
        if emb is None:
            return
        lst = self.staff_body.setdefault(name, [])
        if len(lst) < STAFF_BODY_MAX and all(body.sim(e, emb) < 0.95 for e in lst):
            lst.append(emb)

    def _staff_by_body(self, emb, exclude_names):
        """Bugun yuzi tanilgan xodimga tanasi mos keladimi."""
        if emb is None:
            return None, 0.0
        best, score = None, 0.0
        for name, lst in self.staff_body.items():
            if name in exclude_names:
                continue
            s = max(body.sim(e, emb) for e in lst)
            if s > score:
                best, score = name, s
        return (best, score) if score >= visitors.BODY_THRESHOLD else (None, score)

    # ── asosiy qadam ─────────────────────────────────────────────────
    def step(self, frame, persons, face_results, do_embed):
        """Bir kadr. face_results — arcface.identify natijasi (yoki None).

        do_embed=True bo'lsa tana vektorlari hisoblanadi va hit yig'iladi
        (FACE_INTERVAL bilan bir xil chastota). Qaytaradi:
          found (xodimlar), strangers (mehmonlar), new_visitors, reseen
        """
        active = {p["tid"] for p in persons if "tid" in p}
        self._gc(active)
        new_visitors, reseen = [], []
        if not persons:
            return [], [], new_visitors, reseen
        enabled = visitors.enabled(self.cam)

        if do_embed:
            bodies = body.embed(frame, persons) if enabled else [None] * len(persons)
            # yuzlarni trekka bog'lash (yuz markazi odam qutisida)
            face_of = {}
            for f in face_results or []:
                if f.get("too_small"):
                    continue
                fx = f["box"][0] + f["box"][2] / 2
                fy = f["box"][1] + f["box"][3] / 2
                # Olomonda qutilar ustma-ust: yuz qutining YUQORI yarmida bo'lgan
                # ENG KICHIK qutiga bog'lanadi (katta quti orqadagi odamniki bo'lishi mumkin)
                cands = [p for p in persons if "tid" in p
                         and p["box"][0] <= fx <= p["box"][2]
                         and p["box"][1] <= fy <= p["box"][1] + 0.55 * (p["box"][3] - p["box"][1])]
                if not cands:
                    continue
                p = min(cands, key=lambda q: (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1]))
                if p["tid"] not in face_of or f["box"][2] > face_of[p["tid"]]["box"][2]:
                    face_of[p["tid"]] = f

            dup = _overlapping(persons)
            crowd = _crowded(persons)
            for i, p in enumerate(persons):
                tid = p.get("tid")
                if tid is None or i in dup:
                    continue
                f = face_of.get(tid)
                b = bodies[i]
                crowded = i in crowd    # qo'shni odam kesmaga kiradi: tana saqlanmaydi, chegara qattiq
                g = body.geometry(p)
                # ── 1) xodim yuzi tanildi → xodim ────────────────────
                if f and f["name"]:
                    self.names[tid] = {"name": f["name"], "visitor": None, "n": None,
                                       "box": f["box"], "score": f["score"], "how": "yuz"}
                    self.hits.pop(tid, None)
                    self._remember_staff_body(f["name"], b)
                    continue
                c = self.names.get(tid)
                if c and c.get("name"):
                    # xodim treki — QAYTA TEKSHIRUV: yuz boshqa odamniki bo'lsa darhol uziladi
                    if f and f["name"] and f["name"] != c["name"]:
                        del self.names[tid]
                    elif f and f["name"] is None and f["score"] is not None and f["score"] < visitors.FACE_CONTRA:
                        del self.names[tid]
                        print(f"[kim] {self.cam} trek {tid}: {c['name']} emas — uzildi (yuz {f['score']})")
                    else:
                        if not crowded:
                            self._remember_staff_body(c["name"], b)
                        continue
                if not enabled:
                    continue
                # ── 2) allaqachon mehmonga bog'langan trek → "ko'rildi" ──
                if c and c.get("visitor"):
                    v = visitors.MEMORY.get(c["visitor"])
                    if v is None:
                        del self.names[tid]
                    else:
                        good = bool(f) and visitors.good_face(f)[0] \
                            and f["score"] is not None and f["score"] < visitors.STRANGER_MAX_STAFF
                        # QAYTA TEKSHIRUV — ByteTrack olomonda ID almashtiradi (A raqami B ga o'tadi).
                        # Yuz aniq boshqa → darhol uziladi. Tana ketma-ket 3 marta mos kelmasa → uziladi.
                        fs = visitors.MEMORY._best(v.get("embs"), f["emb"]) if good else None
                        bs = visitors.MEMORY._best(v.get("body_embs"), b) if b is not None else None
                        if fs is not None and fs < visitors.FACE_CONTRA:
                            print(f"[kim] {self.cam} trek {tid}: Begona #{v['n']} emas — uzildi (yuz {fs:.2f})")
                            del self.names[tid]
                        else:
                            if bs is not None and bs < visitors.BODY_UNBIND_MAX and not (fs and fs >= visitors.THRESHOLD):
                                c["strikes"] = c.get("strikes", 0) + 1
                            else:
                                c["strikes"] = 0
                            if c["strikes"] >= visitors.UNBIND_STRIKES:
                                print(f"[kim] {self.cam} trek {tid}: Begona #{v['n']} emas — uzildi (tana {bs:.2f} x{c['strikes']})")
                                del self.names[tid]
                            else:
                                visitors.MEMORY.seen(v, f["emb"] if good else None, self.cam,
                                                     body_emb=None if crowded else b)
                                if f:
                                    c["box"] = f["box"]
                                continue
                    c = None
                # ── 3) hit yig'ish ───────────────────────────────────
                face_emb, staff_score = None, None
                if f and f.get("emb") is not None and f["score"] is not None:
                    staff_score = f["score"]
                    if f["score"] >= visitors.STRANGER_MAX_STAFF:
                        self._diag(tid, f"noaniq xodim bali {f['score']} — yuz hisobga olinmaydi")
                    else:
                        good, why = visitors.good_face(f)
                        if good:
                            face_emb = f["emb"]
                        else:
                            self._diag(tid, f"sifatsiz yuz: {why} — faqat tana")
                if face_emb is None and b is None:
                    self._diag(tid, "na yuz, na tana (quti kichik)")
                    continue
                hs = self.hits.setdefault(tid, [])
                # hitlar BIR odam bo'lsin: yuz bor bo'lsa yuz bilan, aks holda tana bilan
                if hs:
                    prev = hs[-1]
                    if face_emb is not None and prev["face"] is not None:
                        same = float(np.dot(prev["face"], face_emb)) >= visitors.HIT_SAME_MIN
                    elif b is not None and prev["body"] is not None:
                        same = body.sim(prev["body"], b) >= visitors.BODY_HIT_SAME_MIN
                    else:
                        same = True
                    if not same:
                        hs.clear()
                hs.append({"face": face_emb, "body": b, "geom": g, "staff_score": staff_score,
                           "crowded": crowded})
                if len(hs) < visitors.STRANGER_MIN_HITS:
                    self._diag(tid, f"hit {len(hs)}/{visitors.STRANGER_MIN_HITS} "
                                    f"(yuz {'bor' if face_emb is not None else 'yo`q'}, "
                                    f"tana {'bor' if b is not None else 'yo`q'})")
                    continue
                # ── 4) qaror ─────────────────────────────────────────
                faces_ = [h["face"] for h in hs if h["face"] is not None]
                bodies_ = [h["body"] for h in hs if h["body"] is not None]
                face_q = faces_[-1] if faces_ else None
                body_q = None
                if bodies_:
                    body_q = np.mean(bodies_, axis=0)
                    body_q /= (np.linalg.norm(body_q) + 1e-9)
                geom_q = next((h["geom"] for h in reversed(hs) if h["geom"]), None)
                # qoida 1: boshqa faol trekka bog'langan shaxslar nomzod emas
                busy_visitors = {x["visitor"] for t, x in self.names.items()
                                 if t != tid and x.get("visitor")}
                busy_staff = {x["name"] for t, x in self.names.items()
                              if t != tid and x.get("name")}
                # avval: bu xodim emasmi (tana orqali, yuzi bugun tanilgan)?
                if face_q is None:
                    sname, sscore = self._staff_by_body(body_q, busy_staff)
                    if sname:
                        self.names[tid] = {"name": sname, "visitor": None, "n": None,
                                           "box": _head_box(p), "score": round(sscore, 3), "how": "tana"}
                        self.hits.pop(tid, None)
                        print(f"[kim] {self.cam} trek {tid}: xodim {sname} (tana {sscore:.2f})")
                        continue
                in_crowd = any(h.get("crowded") for h in hs)
                v, how, sc = visitors.MEMORY.match(
                    face_q, body_q, geom_q, exclude=busy_visitors,
                    body_thr=visitors.CROWD_BODY_THRESHOLD if in_crowd else None)
                if v is None and face_q is None and in_crowd:
                    self._diag(tid, "olomon, yuz yo'q — raqam berilmaydi, yuz kutiladi")
                    del hs[:-1]
                    continue
                if v is None and face_q is None:
                    # YUZSIZ yangi raqam — faqat aniq holatda:
                    #  (a) tana o'xshashligi noaniq oraliqda emas (aks holda bu o'sha
                    #      odam bo'lishi mumkin — yangi raqam bermaymiz, kutamiz)
                    #  (b) odam tik va yetarli katta (o'tirgan/stol ortidagi tana
                    #      vektori beqaror — yuz kutiladi)
                    bb = visitors.MEMORY.body_best(body_q, exclude=busy_visitors)
                    x1, y1, x2, y2 = p["box"]
                    h, w = y2 - y1, max(1, x2 - x1)
                    if not visitors.BODY_ONLY_REGISTER:
                        self._diag(tid, f"yuzsiz — raqam berilmaydi, yuz kutiladi (tana eng yaqin {bb})")
                        del hs[:-1]
                        continue
                    if bb >= visitors.BODY_AMBIG_MIN:
                        self._diag(tid, f"noaniq tana {bb} (o'sha odam bo'lishi mumkin) — yuz kutiladi")
                        del hs[:-1]
                        continue
                    if h < visitors.BODY_ONLY_MIN_H or h / w < visitors.BODY_ONLY_MIN_ASPECT:
                        self._diag(tid, f"yuzsiz raqam yo'q: quti {w}x{h} (o'tirgan/kichik) — yuz kutiladi")
                        del hs[:-1]
                        continue
                if v is not None:
                    if visitors.MEMORY.seen(v, face_q, self.cam, body_emb=body_q, geom=geom_q):
                        reseen.append(v)
                        print(f"[begona] #{v['n']} qaytdi ({how} {sc}) — {self.cam}")
                    else:
                        print(f"[begona] #{v['n']} o'sha ({how} {sc}) — {self.cam}")
                else:
                    v = visitors.MEMORY.add(face_q, self.cam, body_emb=body_q, geom=geom_q)
                    for h in hs[:-1]:
                        visitors.MEMORY.seen(v, h["face"], body_emb=h["body"])
                    new_visitors.append(v)
                    visitors.save_thumb(v, frame, p["box"])
                    print(f"[begona] YANGI #{v['n']} — {self.cam} "
                          f"(yuz {'bor' if face_q is not None else 'yo`q'}, tana {'bor' if body_q is not None else 'yo`q'})")
                self.names[tid] = {"name": None, "visitor": v["id"], "n": v["n"],
                                   "box": f["box"] if f else _head_box(p), "score": sc, "how": how}
                self.hits.pop(tid, None)

        # ── chiqish (har kadr keshdan) ───────────────────────────────
        found, strangers = [], []
        for p in persons:
            c = self.names.get(p.get("tid"))
            p["name"] = c["name"] if c else None
            p["visitor"] = c.get("n") if c and c.get("visitor") else None
            if c and c["name"]:
                found.append({"box": c["box"], "name": c["name"], "how": c.get("how"),
                              "score": c["score"], "too_small": False})
            elif c:
                strangers.append({"box": c["box"], "name": f"Begona #{c['n']}",
                                  "how": c.get("how"), "score": c["score"], "too_small": False})
        return found, strangers, new_visitors, reseen
