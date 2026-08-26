"""MARS audit kamerasi — qoidalar bo'yicha kuzatuv.

Kamera MARS ning rasmiy audit qoidalarini kuzatadi. Qoidalar bu kodda
yozilmagan — ular Mars API'dan keladi (rules.py), metodist adminda o'zgartirsa
kamera ham o'zgaradi.

Qismlari:
    rules.py      qoidalar (API)          detectors.py  qoida detektorlari
    schedule.py   dars jadvali (API)      faces.py      kim ekanini tanish
    nvr.py        kamera oqimi            pose.py       tana holati

Ishga tushirish:
    BRANCH=Yunusobod ./venv/bin/python app.py
    brauzer: http://localhost:5001

Muhim: detektor JARIMA QO'YMAYDI. U nomzod hodisa ko'rsatadi, qarorni
auditor qabul qiladi — noto'g'ri jarima yo'q signaldan qimmatroq.
"""
import io
import time
import threading
from collections import deque

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template_string,
                   request, send_file)

import attendance
import nvr
import rules
import faces
import pose
import detectors

PORT = 5001
MAX_EVENTS = 60
ANALYZE_INTERVAL = 5.0      # fon kamerasi: grid muzlatilgan, sanoq "Sanash"
                            # tugmasi bilan yangilanadi — tez-tez tahlil shart emas
# Ochilgan kamera TEZ-TEZ tahlil qilinadi — ramkalar odam bilan birga
# yurishi kerak. Yuruvchi odamda 3 sekundlik oraliqda ramka orqada qolardi.
#
# Narx o'lchandi: pose + odam-detektori birga 117 ms. Ya'ni 0.5 sekundda
# bir marta ~23% yuk — kadr yetkazishga sezilarli ta'sir qilmaydi.
FOCUS_ANALYZE_INTERVAL = 0.3

app = Flask(__name__)

EVENTS = deque(maxlen=MAX_EVENTS)
_events_lock = threading.Lock()
_event_seq = [0]
_shots = {}                # hodisa rasmi (id -> jpeg)


def add_event(event, jpeg):
    with _events_lock:
        _event_seq[0] += 1
        event = dict(event, id=_event_seq[0])
        EVENTS.appendleft(event)
        _shots[event["id"]] = jpeg
        # eskilarini tozalash — rasm xotirada yig'ilib qolmasin
        alive = {e["id"] for e in EVENTS}
        for eid in [k for k in _shots if k not in alive]:
            _shots.pop(eid, None)
    return event


def draw(frame, state):
    """Kadr ustiga tahlil natijasini chizadi."""
    for f in state.get("faces", []):
        x, y, w, h = f["box"]
        named = f["name"] is not None
        color = (0, 200, 0) if named else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f["name"] or "?", (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for p in state.get("persons", []):
        if not p["reliable"]:
            continue
        x1, y1, x2, y2 = p["box"]
        if p["head_down"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "bosh pastda", (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for i, ev in enumerate(state.get("events", [])[:3]):
        cv2.putText(frame, f"{ev['rule_number']} {ev['rule_type']}",
                    (20, 40 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    return frame


# Modellar ASOSIY IPDA, KETMA-KET yuklanadi — iplar boshlanishidan oldin.
#
# Ilgari ular analizator ipida yuklanardi, ayni paytda Mac kamerasi ipi
# AVFoundation ni ishga tushirardi. Natijada MetalPerformanceShadersGraph
# (Apple GPU) va CoreML bir vaqtda initsializatsiya qilib, xotira buzilgan
# va Python SIGABRT bilan qulagan (crash hisoboti: malloc_zone_error ->
# abort, com.apple.coreml.MLModelAssetResourceFactory.modelLoadQueue).
print("Modellar yuklanmoqda…")
pose.model()             # yolov8s-pose  (PyTorch / MPS)
pose.detect_model()      # yolov8m       (PyTorch / MPS)
faces.load_models()      # YuNet + SFace (OpenCV DNN / CoreML)
print("Modellar tayyor.")

nvr.build()                      # filiallarni yaratadi va ishga tushiradi
CAMERAS = nvr.all_cameras()      # {'Filial/kanal': Camera}
_last_analyzed = {}


def analyzer():
    """Bitta GPU — kameralarni navbat bilan tahlil qiladi, ochilgani birinchi.

    Barcha filiallar bitta navbatda: GPU bitta, ikkita analizator ip ochish
    faqat bir-birini kutishga olib keladi.
    """
    order = list(CAMERAS)      # modellar allaqachon yuklangan (yuqorida)
    i = 0
    while True:
        # Ochilgan kamera MUTLAQ ustunlikda. Uning kadri hali kelmagan
        # bo'lsa ham fon kamerasiga O'TMAYMIZ — biroz kutamiz. Aks holda
        # 170 ms lik fon tahlillari orasida ochilgan kamera sekundlab
        # navbat kutardi (o'lchandi: ramka yoshi 3.3 sekundgacha).
        watched_cam = None
        for br in nvr.BRANCHES.values():
            ch = br.focused_channel()
            if ch:
                watched_cam = br.cameras[ch]
                break

        cam = None
        if watched_cam is not None:
            since = time.time() - _last_analyzed.get(watched_cam.key, 0)
            if since >= FOCUS_ANALYZE_INTERVAL:
                if watched_cam._raw is not None:
                    cam = watched_cam
                else:
                    time.sleep(0.03)     # kadri kelishini kutamiz, fonni emas
                    continue

        if cam is None:
            # Fon kamerasi — lekin keyingi fokus tahliligacha ULGURSAKKINA.
            # Fon tahlili ~170 ms; fokusga 0.2 sekunddan kam qolgan bo'lsa
            # boshlamaymiz, aks holda fokus kechikadi.
            if watched_cam is not None:
                left = FOCUS_ANALYZE_INTERVAL - (
                    time.time() - _last_analyzed.get(watched_cam.key, 0))
                if left < 0.25:
                    time.sleep(max(0.0, left))
                    continue
            for _ in range(len(order)):
                cand = CAMERAS[order[i % len(order)]]
                i += 1
                if cand is watched_cam:
                    continue
                if time.time() - _last_analyzed.get(cand.key, 0) < ANALYZE_INTERVAL:
                    continue
                if cand._raw is None:
                    continue
                cam = cand
                break
            if cam is None:
                time.sleep(0.05)
                continue

        frame = cam.take_frame()
        if frame is None:
            continue
        _last_analyzed[cam.key] = time.time()

        branch = cam.branch.name
        try:
            persons = pose.people_in(frame)
            # Yuz qidirish eng qimmat qadam (138 ms). Xonada odam bo'lmasa
            # qidirishning ma'nosi yo'q — bo'sh xonalarda bekorga sarflanardi.
            #
            # Shart ODAM BORLIGIGA bog'liq, "holati o'qiladimi" ga emas:
            # yaqindan turgan odamning qutisi kadr chetiga tegadi va
            # reliable=False bo'ladi — Mac kamerasida aynan shu sababli yuz
            # umuman qidirilmasdi.
            found = faces.identify(frame, branch=branch) if persons else []
        except Exception as e:
            print(f"[analyzer] {cam.key} tahlil xatosi: {e}")
            continue

        for f in found:
            if f["name"]:
                attendance.record(f["name"], branch, cam.name)

        ctx = detectors.Context(branch=branch, channel=cam.channel,
                                camera_name=cam.name, faces=found, persons=persons)
        events = detectors.run(ctx)

        h, w = frame.shape[:2]
        # Ramkalar FOIZDA saqlanadi: brauzer rasmni istalgan o'lchamda
        # ko'rsatadi, foiz esa o'lchamga bog'liq emas.
        # Odam qutilari — asosiysi, chunki asosiy funksiya sanash.
        # Raqamlanadi, ya'ni ekranda sanoqni ko'z bilan tekshirsa bo'ladi.
        boxes = [{"x": round(100 * p["box"][0] / w, 2),
                  "y": round(100 * p["box"][1] / h, 2),
                  "w": round(100 * (p["box"][2] - p["box"][0]) / w, 2),
                  "h": round(100 * (p["box"][3] - p["box"][1]) / h, 2),
                  "label": str(i),
                  "kind": "alert" if (p["reliable"] and p["head_down"]) else "person"}
                 for i, p in enumerate(persons, 1)]
        # Yuz ramkasi FAQAT kim ekani aniqlanganda. Tanib bo'lmaydigan
        # kichik yuzlarga ramka chizish ekranni bekorga to'ldiradi.
        boxes += [{"x": round(100 * f["box"][0] / w, 2),
                   "y": round(100 * f["box"][1] / h, 2),
                   "w": round(100 * f["box"][2] / w, 2),
                   "h": round(100 * f["box"][3] / h, 2),
                   "label": f["name"], "kind": "face"}
                  for f in found if f["name"]]
        cam.apply({"at": time.time(),
                   "faces": found, "persons": persons, "events": events,
                   "zone": ctx.zone, "count": ctx.head_count,
                   "named": ctx.named, "boxes": boxes,
                   "identity": ctx.identity_reliable,
                   "face_px": max((f["box"][2] for f in found), default=0)})

        for ev in events:
            ev["camera"] = cam.name
            ev["channel"] = cam.channel
            ev["branch"] = branch
            # Hodisa rasmi — dalil, shuning uchun ramkalar unga CHIZILADI
            # (jonli ko'rinishdan farqli: u yerda brauzer chizadi).
            vis = draw(frame.copy(), {"faces": found, "persons": persons,
                                      "events": [ev]})
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
            add_event(ev, buf.tobytes() if ok else b"")
            print(f"[hodisa] {branch}/{ev['rule_number']} ({ev['rule_type']}, "
                  f"{ev['score']} ball) — {ev['reason']}")


threading.Thread(target=analyzer, daemon=True).start()


PAGE = r"""
<!doctype html><meta charset=utf-8><title>MARS audit kamerasi</title>
<style>
 :root{--bg:#141210;--panel:#1c1916;--card:#221e1a;--line:#332c24;
   --fg:#ece7dd;--dim:#9a9086;--accent:#4ade80;--accentd:#2d4a2d}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
   display:grid;grid-template-columns:210px 1fr;min-height:100vh}
 /* ── chap menyu ── */
 nav{background:var(--panel);border-right:1px solid var(--line);padding:18px 12px;
   display:flex;flex-direction:column;gap:4px;position:sticky;top:0;height:100vh}
 .logo{font-weight:700;font-size:15px;padding:6px 10px 16px;letter-spacing:.2px}
 .logo small{display:block;color:var(--dim);font-weight:400;font-size:11px;
   letter-spacing:0}
 .navbtn{display:flex;align-items:center;gap:10px;background:none;border:0;
   color:var(--dim);padding:10px 12px;border-radius:8px;cursor:pointer;
   font-size:14px;text-align:left;width:100%}
 .navbtn:hover{background:var(--card);color:var(--fg)}
 .navbtn.act{background:var(--accentd);color:#dff5df}
 .navbtn .ic{font-size:16px;width:18px;text-align:center}
 nav .foot{margin-top:auto;color:var(--dim);font-size:11px;padding:10px;
   line-height:1.5}
 /* ── asosiy maydon ── */
 main{padding:20px 24px;overflow:auto;max-height:100vh}
 .head{display:flex;align-items:center;gap:14px;margin-bottom:18px;flex-wrap:wrap}
 .head h2{font-size:18px;margin:0;font-weight:600}
 .head .sp{flex:1}
 .pill{display:inline-flex;align-items:baseline;gap:5px;background:var(--card);
   border:1px solid var(--line);border-radius:20px;padding:4px 12px;font-size:13px}
 .pill b{font-size:16px}
 .btn{background:var(--card);color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:7px 14px;cursor:pointer;font-size:13px}
 .btn:hover{border-color:#6b5b45}
 .btn:disabled{opacity:.5;cursor:default}
 .btn.go{background:var(--accentd);border-color:#3d6b3d;color:#dff5df}
 select,input[type=text]{background:#14110e;color:var(--fg);
   border:1px solid var(--line);border-radius:8px;padding:8px 11px;font-size:14px}
 .tabseg{display:flex;gap:6px}
 .tabseg button{background:transparent;color:var(--dim);border:1px solid var(--line);
   border-radius:20px;padding:4px 14px;cursor:pointer;font-size:13px}
 .tabseg button.act{background:var(--card);color:var(--fg);border-color:#6b5b45}
 .muted{color:var(--dim);font-size:13px}
 .view{display:none} .view.on{display:block}
 /* ── kameralar ── */
 .camwrap{display:grid;grid-template-columns:1fr 320px;gap:18px;align-items:start}
 @media(max-width:1000px){.camwrap{grid-template-columns:1fr}}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 .cam{background:var(--card);border:1px solid var(--line);border-radius:10px;
   overflow:hidden;cursor:pointer;transition:border-color .15s}
 .cam:hover{border-color:#6b5b45}
 .cam .shot{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#0b0908}
 .cam .body{padding:9px 11px}
 .cam .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
 .cam .nm{font-weight:600;font-size:13px}
 .cam .zone{color:var(--dim);font-size:11px}
 .cam .cnt{display:flex;align-items:baseline;gap:6px;margin-top:5px}
 .cam .num{font-size:26px;font-weight:700;line-height:1}
 .cam .num.zero{color:var(--dim);font-weight:400}
 .cam .unit{color:var(--dim);font-size:12px}
 .cam .idbadge{margin-left:auto;font-size:10px;padding:2px 7px;border-radius:20px}
 .idok{background:var(--accentd);color:#9fd89f} .idno{background:#3a3230;color:#b9a89a}
 .cam.off{opacity:.45} .cam.hit{outline:2px solid #d9534f}
 .side{background:var(--card);border:1px solid var(--line);border-radius:10px;
   padding:14px}
 .side h3{font-size:13px;margin:0 0 10px;color:var(--dim);font-weight:600;
   text-transform:uppercase;letter-spacing:.4px}
 .ev{border-bottom:1px solid var(--line);padding:10px 0}
 .ev:last-child{border:0} .ev b{font-size:13px}
 .ev img{width:100%;border-radius:6px;margin-top:6px}
 .tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;
   font-weight:600;margin-right:6px}
 .green{background:var(--accentd);color:#9fd89f} .yellow{background:#4a432d;color:#e0d18a}
 .red{background:#4a2d2d;color:#efa0a0} .black{background:#3a3a3a;color:#ddd}
 .info{background:#2d3a4a;color:#a8c8e0}
 .empty{color:var(--dim);padding:24px 0;text-align:center}
 /* ── katta ko'rinish ── */
 #big{position:fixed;inset:0;background:#000d;display:none;z-index:20;
   align-items:center;justify-content:center;flex-direction:column;gap:12px}
 #big.on{display:flex}
 #bigwrap{position:relative;display:inline-block;line-height:0}
 #big img{max-width:92vw;max-height:80vh;border-radius:10px;background:#000}
 .ov{position:absolute;border:2px solid;border-radius:3px;pointer-events:none}
 .ov span{position:absolute;top:-19px;left:-2px;font-size:11px;line-height:1.4;
   padding:0 5px;border-radius:3px;white-space:nowrap;color:#111;font-weight:600}
 .ov.person{border-color:var(--accent)} .ov.person span{background:var(--accent)}
 .ov.face{border-color:#60a5fa} .ov.face span{background:#60a5fa}
 .ov.alert{border-color:#f87171} .ov.alert span{background:#f87171}
 #bigbar{color:var(--fg);display:flex;gap:14px;align-items:center;font-size:14px}
 #bigcount{font-size:22px}
 #bigtop{min-height:44px;display:flex;gap:10px;align-items:center;
   justify-content:center;flex-wrap:wrap}
 .faceb{font-size:20px;font-weight:700;padding:8px 20px;border-radius:30px}
 .faceb.ok{background:var(--accent);color:#062b10}
 .faceb.small{background:#4a432d;color:#e0d18a;font-size:15px;font-weight:600}
 /* ── davomat ── */
 table.att{width:100%;border-collapse:collapse;font-size:14px;
   background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
 table.att th{text-align:left;color:var(--dim);font-weight:500;font-size:12px;
   padding:11px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
 table.att td{padding:11px 14px;border-bottom:1px solid var(--line)}
 table.att tr:last-child td{border:0}
 table.att td.t{font-variant-numeric:tabular-nums;font-size:15px}
 .att-cam{color:var(--dim);font-size:11px}
 /* ── xodimlar ── */
 .staffwrap{display:grid;grid-template-columns:380px 1fr;gap:24px;align-items:start}
 @media(max-width:900px){.staffwrap{grid-template-columns:1fr}}
 .enroll{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
 .enroll .fld{margin-bottom:10px}
 .enroll .fld label{display:block;color:var(--dim);font-size:12px;margin-bottom:5px}
 .enroll input,.enroll select{width:100%}
 #preview{width:100%;border-radius:8px;margin:10px 0 0;display:none;
   background:#000;aspect-ratio:4/3;object-fit:cover;transform:scaleX(-1)}
 #npreview{width:100%;border-radius:8px;margin:10px 0 0;display:none;
   background:#000;aspect-ratio:16/9;object-fit:cover}
 #step{margin-top:10px;font-size:15px;font-weight:600;min-height:20px}
 #bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px;
   overflow:hidden;display:none}
 #bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
 #msg{margin-top:10px;font-size:13px;min-height:18px}
 #msg.ok{color:#9fd89f} #msg.err{color:#e08a8a}
 .plist{background:var(--card);border:1px solid var(--line);border-radius:10px;
   overflow:hidden}
 .prow{display:flex;justify-content:space-between;align-items:center;gap:10px;
   padding:11px 14px;border-bottom:1px solid var(--line);font-size:14px}
 .prow:last-child{border:0}
 .prow .n{flex:1;font-weight:500} .prow .s{color:var(--dim);font-size:12px}
 .prow button{padding:4px 11px;font-size:12px;background:transparent;
   border:1px solid var(--line);color:var(--dim);border-radius:6px;cursor:pointer}
 .prow button:hover{border-color:#a05a5a;color:#e0a8a8}
 .prow button.armed{background:#4a2d2d;border-color:#a05a5a;color:#efa0a0}
 .warn{margin-bottom:14px;border-radius:8px;font-size:13px;
   background:#3a2d2a;color:#e0b3a8}
 .warn summary{padding:11px 14px;cursor:pointer;font-weight:600;list-style:none}
 .warn summary::-webkit-details-marker{display:none}
 .warn[open] summary{border-bottom:1px solid #52403c}
 .warn div{padding:12px 14px;line-height:1.7}
</style>
<body>
<nav>
  <div class=logo>MARS<small>audit kamerasi</small></div>
  <button class=navbtn id=nav-cameras onclick="showView('cameras')">
    <span class=ic>▦</span> Kameralar</button>
  <button class=navbtn id=nav-attendance onclick="showView('attendance')">
    <span class=ic>◷</span> Davomat</button>
  <button class=navbtn id=nav-staff onclick="showView('staff')">
    <span class=ic>☺</span> Xodimlar</button>
  <div class=foot id=foot>yuklanmoqda…</div>
</nav>
<main>
  <!-- ═══ KAMERALAR ═══ -->
  <section class="view on" id=view-cameras>
    <div class=head>
      <h2>Kameralar</h2>
      <span class=tabseg id=tabs></span>
      <div class=sp></div>
      <span class=pill><b id=total>0</b> odam</span>
      <button class=btn id=scanbtn onclick="doScan()">Sanash</button>
    </div>
    <div class=camwrap>
      <div class=grid id=grid></div>
      <div class=side>
        <h3>Hodisalar</h3>
        <div id=events><div class=empty>hozircha yo'q</div></div>
      </div>
    </div>
  </section>
  <!-- ═══ DAVOMAT ═══ -->
  <section class=view id=view-attendance>
    <div class=head>
      <h2>Davomat</h2>
      <div class=sp></div>
      <select id=attdate onchange="loadAtt()"></select>
    </div>
    <p class=muted style="margin-top:-6px;max-width:640px">
      Keldi = shu kuni birinchi tanilgan vaqt · Ketdi = oxirgi tanilgan vaqt.
      Tanish faqat yuz katta ko'rinadigan kameralarda ishlaydi.</p>
    <div id=atttable></div>
  </section>
  <!-- ═══ XODIMLAR ═══ -->
  <section class=view id=view-staff>
    <div class=head><h2>Xodimlar</h2></div>
    <div class=staffwrap>
      <div class=enroll>
        <div class=fld>
          <label>Ism familiya</label>
          <input type=text id=pname placeholder="masalan: Sardor Madaliyev" autocomplete=off>
        </div>
        <div class=fld>
          <label>Qaysi kameradan</label>
          <select id=psrc></select>
        </div>
        <video id=preview autoplay muted playsinline></video>
        <img id=npreview>
        <div id=step></div>
        <div id=bar><i></i></div>
        <button class="btn go" id=addbtn onclick="addFace()"
          style="width:100%;margin-top:12px;padding:10px">Yuzni olish</button>
        <div id=msg></div>
        <p class=muted style="margin-top:12px;line-height:1.6">
          Yuz turli burchakdan olinadi — panel yo'l-yo'riq beradi. Bir odamni
          bir necha marta olsa tanish yaxshilanadi. Rasm saqlanmaydi.</p>
      </div>
      <div>
        <div id=pwarn></div>
        <div class=plist id=plist></div>
      </div>
    </div>
  </section>
</main>
<div id=big><div id=bigtop></div><div id=bigwrap><img id=bigimg></div><div id=bigbar>
  <span id=bigname></span><b id=bigcount>0</b><span class=muted>odam</span>
  <span class=muted id=bigfps></span><span class=muted id=bigage></span>
  <button class=btn onclick="closeBig()">Yopish (Esc)</button></div></div>
<script>
let branch=null, built=false, view="cameras";
function showView(v){
  view=v;
  for(const x of ["cameras","attendance","staff"]){
    document.getElementById("view-"+x).classList.toggle("on", x===v);
    document.getElementById("nav-"+x).classList.toggle("act", x===v);
  }
  if(v!=="cameras") closeBig();
  if(v!=="staff") closeCam();
  if(v==="attendance") loadAtt();
  if(v==="staff") loadPeople();
}
// ── kameralar ──
function buildTabs(list){
  const box=document.getElementById("tabs");
  if(box.childElementCount===list.length) return;
  box.innerHTML="";
  for(const b of list){
    const t=document.createElement("button");
    t.textContent=b;
    t.onclick=()=>{ if(branch===b) return;
      branch=b; built=false; closeBig();
      document.getElementById("grid").innerHTML=""; tick(); };
    box.appendChild(t);
  }
}
function build(cams){
  const grid=document.getElementById("grid"); grid.innerHTML="";
  for(const c of cams){
    const d=document.createElement("div"); d.className="cam"; d.id="c"+c.channel;
    d.innerHTML=`<img class=shot id="s${c.channel}">
      <div class=body>
        <div class=top><span class=nm>${c.name}</span>
          <span class=zone>${c.zone||""}</span></div>
        <div class=cnt><span class=num id="n${c.channel}">0</span>
          <span class=unit>odam</span>
          <span class="idbadge idno" id="b${c.channel}">—</span></div>
      </div>`;
    d.onclick=()=>openBig(c.channel,c.name);
    grid.appendChild(d);
  }
  built=true;
}
async function doScan(){
  const b=document.getElementById("scanbtn");
  b.disabled=true; b.textContent="Sanalyapti…";
  await fetch("/scan/"+encodeURIComponent(branch),{method:"POST"});
}
// ── katta ko'rinish ──
let bigCh=null, lastFrameAt=0, frameGen=0, lastUrl=null;
const BOX_MAX_AGE=1.2;
async function frameLoop(br, ch, gen){
  const img=document.getElementById("bigimg"); let seq=-1;
  while(bigCh===ch && frameGen===gen){
    try{
      const r=await fetch(`/frame/${encodeURIComponent(br)}/${ch}?after=${seq}&big=1`);
      if(r.status===204) continue;
      if(!r.ok){ await new Promise(s=>setTimeout(s,400)); continue; }
      seq=+r.headers.get("X-Seq");
      const url=URL.createObjectURL(await r.blob());
      img.src=url; if(lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl=url; lastFrameAt=Date.now();
      // Ramkalarni KADR BILAN BIRGA chizamiz — tick (3s) ni kutmaymiz.
      try{
        const m=JSON.parse(r.headers.get("X-Meta")||"{}");
        const fresh=m.age!=null && m.age<=BOX_MAX_AGE;
        drawBoxes(fresh ? (m.boxes||[]) : []);
        document.getElementById("bigcount").textContent=m.count||0;
        const top=document.getElementById("bigtop");
        if(m.named && m.named.length)
          top.innerHTML=m.named.map(n=>`<span class="faceb ok">✓ ${n}</span>`).join("");
        else if(m.face_px>=25)
          top.innerHTML=`<span class="faceb small">Yuz topildi (${m.face_px}px) — tanish uchun yaqinroq keling</span>`;
        else top.innerHTML="";
      }catch(e){}
    }catch(e){ await new Promise(s=>setTimeout(s,400)); }
  }
}
function drawBoxes(boxes){
  const wrap=document.getElementById("bigwrap"), img=document.getElementById("bigimg");
  for(const el of [...wrap.querySelectorAll(".ov")]) el.remove();
  if(!img.clientWidth) return;
  for(const b of boxes){
    const d=document.createElement("div"); d.className="ov "+b.kind;
    d.style.left=b.x+"%"; d.style.top=b.y+"%";
    d.style.width=b.w+"%"; d.style.height=b.h+"%";
    d.innerHTML="<span>"+b.label+"</span>"; wrap.appendChild(d);
  }
}
function openBig(ch,name){
  bigCh=ch; lastFrameAt=0; drawBoxes([]);
  document.getElementById("bigimg").removeAttribute("src");
  frameLoop(branch, ch, ++frameGen);
  document.getElementById("bigname").textContent=name;
  document.getElementById("big").classList.add("on");
}
function closeBig(){
  bigCh=null; frameGen++; lastFrameAt=0; drawBoxes([]);
  document.getElementById("bigimg").removeAttribute("src");
  document.getElementById("big").classList.remove("on");
}
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeBig();});
// ── davomat ──
async function loadAtt(){
  const sel=document.getElementById("attdate");
  const q=sel.value?("?date="+sel.value):"";
  const d=await (await fetch("/attendance"+q)).json();
  if(!sel.options.length || sel.options.length!==d.dates.length){
    const cur=sel.value||d.date;
    sel.innerHTML=d.dates.length
      ? d.dates.map(x=>`<option ${x===cur?"selected":""}>${x}</option>`).join("")
      : `<option>${d.date}</option>`;
  }
  document.getElementById("atttable").innerHTML = d.rows.length ? `
    <table class=att><tr><th>Xodim</th><th>Keldi</th><th>Ketdi</th>
      <th>Kamera</th><th></th></tr>
    ${d.rows.map(r=>`<tr>
      <td><b>${r.name}</b></td>
      <td class=t>${r.first.slice(0,5)}</td>
      <td class=t>${r.last.slice(0,5)}</td>
      <td class=att-cam>${r.last_cam}</td>
      <td class=att-cam>${r.seen}x</td></tr>`).join("")}
    </table>`
    : "<div class=empty>Bu kunda yozuv yo'q</div>";
}
// ── xodimlar ──
async function loadSources(){
  const sel=document.getElementById("psrc");
  if(sel.options.length) return;
  const opts=["<option value=mac>Mac kamera (brauzer)</option>"];
  for(const b of (await (await fetch("/state")).json()).branches){
    if(b==="Mac") continue;
    const s=await (await fetch("/state?branch="+encodeURIComponent(b))).json();
    for(const c of s.cameras)
      opts.push(`<option value="${b}/${c.channel}">${b} — ${c.name}</option>`);
  }
  sel.innerHTML=opts.join("");
}
async function loadPeople(){
  loadSources();
  const d=await (await fetch("/faces")).json();
  const box=document.getElementById("pwarn");
  if(d.problems.length){
    const items=d.problems.map(p=> p.type==="duplicate"
      ? `${p.a} va ${p.b} — bir odammi? (${p.score})`
      : `${p.name}: namunalar aralashgan (${p.samples} ta)`).join("<br>");
    box.innerHTML=`<details class=warn><summary>${d.problems.length} ta muammo — bazani tekshiring</summary><div>${items}</div></details>`;
  } else box.innerHTML="";
  document.getElementById("plist").innerHTML = d.people.map((p,i)=>`
    <div class=prow><span class=n>${p.name}</span>
      <span class=s>${p.samples} namuna</span>
      <button class=del data-name="${p.name.replace(/"/g,"&quot;")}"
        onclick="askDel(this)">O'chirish</button>
    </div>`).join("") || "<div class=empty>Bazada xodim yo'q</div>";
}
function say(t,ok){ const m=document.getElementById("msg");
  m.textContent=t; m.className=ok?"ok":"err"; }
const STEPS=[["To'g'riga qarang",3],["Sekin CHAPGA buring",3],
             ["Sekin O'NGGA buring",3],["Biroz YUQORIGA",2],["Biroz PASTGA",2]];
let camStream=null;
async function openCam(){
  if(!camStream) camStream=await navigator.mediaDevices.getUserMedia(
    {video:{width:{ideal:1280},height:{ideal:960}}});
  return camStream;
}
function closeCam(){
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
}
function grab(v){
  const c=document.createElement("canvas");
  c.width=v.videoWidth; c.height=v.videoHeight;
  c.getContext("2d").drawImage(v,0,0);
  return new Promise(r=>c.toBlob(r,"image/jpeg",0.92));
}
async function addFace(){
  const name=document.getElementById("pname").value.trim();
  if(!name){ say("Ismni yozing", false); return; }
  const src=document.getElementById("psrc").value;
  if(src!=="mac"){ return addFaceFromNvr(name, src); }
  const btn=document.getElementById("addbtn");
  const vid=document.getElementById("preview"), bar=document.getElementById("bar");
  const step=document.getElementById("step"), fill=bar.querySelector("i");
  btn.disabled=true; say("", true);
  try{
    vid.srcObject=await openCam(); vid.style.display="block"; bar.style.display="block";
    await new Promise(r=>{ if(vid.videoWidth) r(); else vid.onloadedmetadata=r; });
  }catch(e){ btn.disabled=false; say("Kameraga ruxsat berilmadi: "+e.message, false); return; }
  const total=STEPS.reduce((a,s)=>a+s[1],0);
  let done=0, saved=0, skipped=0, lastErr="";
  for(const [text,shots] of STEPS){
    step.textContent=text;
    await new Promise(s=>setTimeout(s,1300));
    for(let i=0;i<shots;i++){
      const blob=await grab(vid);
      const fd=new FormData(); fd.append("name",name); fd.append("image",blob,"f.jpg");
      try{
        const d=await (await fetch("/faces/image",{method:"POST",body:fd})).json();
        if(d.ok) saved++; else { skipped++; lastErr=d.message; }
      }catch(e){ skipped++; lastErr=e.message; }
      done++; fill.style.width=(done/total*100)+"%";
      step.textContent=text+"  ("+saved+" ta olindi)";
      await new Promise(s=>setTimeout(s,420));
    }
  }
  closeCam(); vid.srcObject=null; vid.style.display="none"; bar.style.display="none";
  step.textContent=""; fill.style.width="0"; btn.disabled=false;
  if(saved) say(name+": "+saved+" ta namuna saqlandi"+(skipped?" ("+skipped+" o'tkazildi)":""), true);
  else say(lastErr || "Yuz olinmadi", false);
  document.getElementById("pname").value = saved ? "" : name;
  loadPeople();
}
async function addFaceFromNvr(name, src){
  const [br, ch]=src.split("/");
  const btn=document.getElementById("addbtn"), bar=document.getElementById("bar");
  const step=document.getElementById("step"), fill=bar.querySelector("i");
  const img=document.getElementById("npreview");
  btn.disabled=true; bar.style.display="block"; img.style.display="block"; say("", true);
  let alive=true, seq=-1;
  (async()=>{ while(alive){
    try{
      const r=await fetch(`/frame/${encodeURIComponent(br)}/${ch}?after=${seq}&big=1`);
      if(r.status===204) continue;
      if(!r.ok){ await new Promise(s=>setTimeout(s,400)); continue; }
      seq=+r.headers.get("X-Seq");
      const u=URL.createObjectURL(await r.blob());
      const old=img.src; img.src=u; if(old.startsWith("blob:")) URL.revokeObjectURL(old);
    }catch(e){ await new Promise(s=>setTimeout(s,400)); }
  }})();
  const total=STEPS.reduce((a,s)=>a+s[1],0);
  let done=0, saved=0, skipped=0, lastErr="";
  for(const [text,shots] of STEPS){
    step.textContent=text;
    await new Promise(s=>setTimeout(s,1600));
    for(let i=0;i<shots;i++){
      try{
        const d=await (await fetch("/faces",{method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({name, branch:br, channel:ch, branches:[br]})})).json();
        if(d.ok) saved++; else { skipped++; lastErr=d.message; }
      }catch(e){ skipped++; lastErr=e.message; }
      done++; fill.style.width=(done/total*100)+"%";
      step.textContent=text+"  ("+saved+" ta olindi)";
      await new Promise(s=>setTimeout(s,600));
    }
  }
  alive=false; img.style.display="none"; img.removeAttribute("src");
  bar.style.display="none"; step.textContent=""; fill.style.width="0"; btn.disabled=false;
  if(saved) say(name+": "+saved+" ta namuna saqlandi"+(skipped?" ("+skipped+" o'tkazildi)":""), true);
  else say(lastErr || "Yuz olinmadi", false);
  document.getElementById("pname").value = saved ? "" : name;
  loadPeople();
}
let delArmed=null, delTimer=null;
function askDel(btn){
  const name=btn.dataset.name;
  if(delArmed===name){                    // ikkinchi bosish — o'chiramiz
    clearTimeout(delTimer); delArmed=null;
    delFace(name); return;
  }
  // birinchi bosish — tasdiq so'raymiz
  document.querySelectorAll(".del").forEach(resetDel);
  delArmed=name; btn.textContent="Tasdiqlang?"; btn.classList.add("armed");
  delTimer=setTimeout(()=>{ resetDel(btn); delArmed=null; }, 3000);
}
function resetDel(btn){ btn.textContent="O'chirish"; btn.classList.remove("armed"); }
async function delFace(name){
  const d=await (await fetch("/faces/"+encodeURIComponent(name),{method:"DELETE"})).json();
  loadPeople();
}
// ── har 3 sekundda holat ──
async function tick(){
  const s=await (await fetch("/state"+(branch?"?branch="+encodeURIComponent(branch):""))).json();
  branch=s.branch;
  buildTabs(s.branches);
  for(const t of document.getElementById("tabs").children)
    t.classList.toggle("act", t.textContent===branch);
  if(!built) build(s.cameras);
  document.getElementById("total").textContent=s.cameras.reduce((a,c)=>a+c.count,0);
  const warn = s.locked ? `NVR QULFLANGAN — ${Math.ceil(s.lock_left/60)} daqiqa`
             : (!s.reachable ? "NVR ga ulanmadi" : "");
  const b=document.getElementById("scanbtn");
  b.disabled=s.scanning; b.textContent=s.scanning?"Sanalyapti…":"Sanash";
  const ago = s.scanned_ago==null ? "" :
    (s.scanned_ago<60 ? `${s.scanned_ago}s oldin` : `${Math.floor(s.scanned_ago/60)} daq oldin`);
  document.getElementById("foot").innerHTML =
    (warn ? `<span style=color:#e08a8a>${warn}</span><br>` : "") +
    `${s.online}/${s.cameras.length} kamera<br>` +
    (ago ? `sanoq ${ago}<br>` : "") +
    `${s.rules} qoida · ${s.detectors} detektor`;
  for(const c of s.cameras){
    const el=document.getElementById("c"+c.channel); if(!el) continue;
    el.classList.toggle("hit", c.hit); el.classList.toggle("off", !c.online);
    const n=document.getElementById("n"+c.channel);
    n.textContent=c.count; n.classList.toggle("zero", c.count===0);
    const bd=document.getElementById("b"+c.channel);
    bd.textContent=c.identity?"yuz aniq":"yuz kichik";
    bd.className="idbadge "+(c.identity?"idok":"idno");
    if(bigCh===null && (s.local || lastScan!==s.scanned_ago))
      document.getElementById("s"+c.channel).src=
        "/still/"+encodeURIComponent(branch)+"/"+c.channel+"?t="+Date.now();
    if(c.channel===bigCh){
      document.getElementById("bigfps").textContent=c.fps+" kadr/sek";
      const live=Date.now()-lastFrameAt<1500;
      document.getElementById("bigage").textContent=live?"":"kadr kelmayapti…";
      if(!live) drawBoxes([]);        // kadr kelmasa ramka ham yo'q
    }
  }
  lastScan=s.scanned_ago;
  const evbox=document.getElementById("events");
  evbox.innerHTML = s.events.length ? s.events.map(e=>`
    <div class=ev>
      <span class="tag ${e.rule_type}">${e.rule_number==="LOKAL"?"Ogohlantirish":e.rule_number+" · "+e.score+" ball"}</span>
      <b>${e.camera}</b>
      <div class=muted>${e.at.replace("T"," ")}</div>
      <div>${e.reason}</div>
      <div class=muted style="margin-top:4px">${e.rule_text}</div>
      <img src="/shot/${e.id}" loading=lazy>
    </div>`).join("") : "<div class=empty>hodisa yo'q</div>";
  if(view==="attendance") loadAtt();
}
// URL hash bilan bo'lim ochish (#davomat, #xodimlar)
const hashView={davomat:"attendance",xodimlar:"staff",kameralar:"cameras"};
if(hashView[location.hash.slice(1)]) showView(hashView[location.hash.slice(1)]);
tick(); setInterval(tick,3000);
</script>
"""



@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/state")
def state():
    """Bitta filial holati. ?branch=Nomi — qaysi filial (birinchisi standart)."""
    name = (request.args.get("branch")
            or (nvr.ENABLED[0] if nvr.ENABLED else next(iter(nvr.BRANCHES), "")))
    br = nvr.BRANCHES.get(name)
    with _events_lock:
        events = [e for e in EVENTS if e.get("branch") == name][:20]
    hits = {e["channel"] for e in events[:6]}
    cams = []
    for cam in (br.cameras.values() if br else []):
        st = cam.state
        cams.append({
            "channel": cam.channel, "name": cam.name, "online": cam.online,
            "zone": detectors.ZONES.get(name, {}).get(cam.channel),
            "count": st.get("count", 0), "named": st.get("named", []),
            "identity": st.get("identity", False),
            "boxes": st.get("boxes", []),
            "boxes_age": round(time.time() - st["at"], 1) if st.get("at") else None,
            "fps": round(cam.fps, 1),
            "face_px": st.get("face_px", 0),
            "hit": cam.channel in hits,
        })
    done, _ = detectors.status()
    locked, left = br.lock_state() if br else (False, 0)
    return jsonify(branch=name, branches=list(nvr.BRANCHES),
                   local=bool(br and getattr(br, "local", False)),
                   scanning=bool(br and br.scanning),
                   scanned_ago=int(time.time() - br.scanned_at)
                   if br and br.scanned_at else None,
                   reachable=(br.reachable is not False) if br else False,
                   cameras=cams, events=events,
                   online=sum(1 for c in cams if c["online"]),
                   locked=locked, lock_left=left,
                   rules=len(rules.load()), detectors=len(done))


@app.post("/scan/<branch>")
def scan(branch):
    """Bitta sanash aylanishi. Grid muzlatilgan, sanoq shu tugma bilan."""
    br = nvr.BRANCHES.get(branch)
    if br is None:
        return jsonify(ok=False), 404
    br.request_scan()
    return jsonify(ok=True)


@app.get("/still/<branch>/<channel>")
def still(branch, channel):
    """Bitta kadr. Grid shuni ishlatadi — oqim EMAS.

    Ilgari grid 15 ta MJPEG oqimini bir vaqtda ochardi. Har oqim Flask ipini
    doimiy band qilib, server bo'g'ilib qolardi: skrinshotda hamma kamera
    qop-qora edi va /state hammasini "offline" deb ko'rsatardi. Bitta kadr
    so'rovi ulanishni ushlab turmaydi.
    """
    cam = nvr.find(branch, channel)
    if cam is None:
        return "yo'q", 404
    return Response(cam.snapshot(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ── Xodimlarni ro'yxatga olish ───────────────────────────────────────
@app.get("/attendance")
def attendance_get():
    """Kun davomati. ?date=YYYY-MM-DD — istalgan kun (standart: bugun)."""
    d, data = attendance.day(request.args.get("date"))
    rows = [dict(name=n, **e) for n, e in data.items()]
    rows.sort(key=lambda r: r["first"])
    return jsonify(date=d, rows=rows, dates=attendance.days())


@app.get("/faces")
def faces_list():
    return jsonify(people=faces.people(), branches=list(nvr.BRANCHES),
                   min_px=faces.MIN_RECOGNIZE_PX,
                   enroll_px=faces.MIN_ENROLL_PX,
                   problems=faces.audit())


@app.post("/faces")
def faces_add():
    """Kameradan yuz olib, ismga yozadi.

    Kadr AYNI DAMDAGI kameradan olinadi — alohida rasm saqlanmaydi.
    Mac kamerasi tavsiya etiladi: NVR kameralarida yuz 13-41 piksel
    bo'ladi va ro'yxatga olishga yaramaydi.
    """
    body = request.get_json(silent=True) or {}
    name = body.get("name", "")
    branch = body.get("branch") or "Mac"
    channel = body.get("channel") or "0"
    cam = nvr.find(branch, channel)
    if cam is None:
        return jsonify(ok=False, message="Kamera topilmadi"), 404
    if hasattr(cam.branch, "want"):
        cam.branch.want()
        # Kamera endi yoqilgan bo'lishi mumkin — birinchi kadrni kutamiz
        for _ in range(30):
            if cam.jpeg is not None:
                break
            time.sleep(0.1)
    frame = cam.take_frame()
    if frame is None:
        # take_frame bir marta beradi — tahlil olib qo'ygan bo'lishi mumkin
        data = cam.snapshot()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, message="Kadr olinmadi"), 503
    ok, msg = faces.enroll(frame, name, branches=body.get("branches"))
    return jsonify(ok=ok, message=msg)


@app.post("/faces/image")
def faces_add_image():
    """Brauzer yuborgan rasmdan yuz oladi.

    Kamera BRAUZERDA ochiladi (getUserMedia) — 5002 dagidek. Sabab:
    serverdagi kamera ko'rinishi tarmoq orqali kechikadi va odam ko'rgan
    kadri bilan saqlangan kadr bir xil bo'lmaydi. Brauzerda esa ko'rinish
    ham, olingan kadr ham aynan bitta manbadan.
    """
    name = request.form.get("name", "").strip()
    photo = request.files.get("image")
    if not photo:
        return jsonify(ok=False, message="Rasm kelmadi"), 400
    raw = np.frombuffer(photo.read(), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, message="Rasm o'qilmadi"), 400
    branches = request.form.get("branches")
    ok, msg = faces.enroll(frame, name,
                           branches=branches.split(",") if branches else None)
    return jsonify(ok=ok, message=msg)


@app.delete("/faces/<path:name>")
def faces_delete(name):
    ok, msg = faces.remove(name)
    return jsonify(ok=ok, message=msg)


@app.get("/frame/<branch>/<channel>")
def frame(branch, channel):
    """Bitta YANGI kadrni kutib qaytaradi (long-poll).

    Nega MJPEG emas: brauzer MJPEG oqimi jimgina to'xtaganda hech qanday
    xato bermaydi — rasm qorayadi, JS esa buni bilmaydi va eski ramkalarni
    chizishda davom etadi. Sardor ko'rgan "qora ekran + ramkalar" aynan shu.
    Bu yerda esa har kadr alohida javob: kelmasa mijoz darhol biladi.

    ?after=N — mijozdagi oxirgi kadr raqami. Undan yangisi chiqguncha
    kutamiz (WAIT gacha), keyin qaytaramiz.
    """
    cam = nvr.find(branch, channel)
    if cam is None:
        return "yo'q", 404
    after = request.args.get("after", type=int, default=-1)
    if hasattr(cam.branch, "want"):
        cam.branch.want()
    if request.args.get("big") == "1":
        cam.branch.set_focus(channel)
    WAIT = 3.0
    deadline = time.time() + WAIT
    while cam.seq == after and time.time() < deadline:
        time.sleep(0.02)
    if cam.seq == after:
        return Response(status=204)      # yangi kadr yo'q
    with cam.lock:
        data, seq = cam.jpeg, cam.seq
        st = dict(cam.state)
    import json as _json
    named = [f["name"] for f in st.get("faces", []) if f.get("name")]
    facepx = max((f["box"][2] for f in st.get("faces", [])), default=0)
    meta = _json.dumps({"boxes": st.get("boxes", []),
                        "count": st.get("count", 0),
                        "named": named, "face_px": facepx,
                        "age": round(time.time() - st["at"], 1) if st.get("at") else None})
    return Response(data or nvr.PLACEHOLDER, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Seq": str(seq),
                             "X-Meta": meta})


@app.get("/stream/<branch>/<channel>")
def stream(branch, channel):
    cam = nvr.find(branch, channel)
    if cam is None:
        return "yo'q", 404
    big = request.args.get("big") == "1"
    token = cam.branch.new_stream_token() if big else None

    def gen():
        last = -1
        while True:
            # Kimdir katta ko'rinishda qarab turibdi — fokusni ushlab turamiz.
            # Ilgari fokus faqat bosilganda yoqilardi va TTL 6 sekundda
            # so'nardi, ya'ni amalda hech qachon ishlamasdi: 15 kamera
            # birdek sekin so'ralib, hammasi 0.7 kadr/sek edi.
            # Faqat ENG OXIRGI so'rov fokusni belgilaydi. Eskisi oqimni
            # davom ettiradi, lekin fokusni tortmaydi — aks holda kamera
            # almashtirilganda ikkisi navbatlashib, ekran qorayardi.
            #
            # Eskisini butunlay to'xtatib bo'lmaydi: bir vaqtda ikki kishi
            # qarashi mumkin va ikkinchisi birinchisini o'chirib qo'ymasin.
            if big and token == cam.branch.stream_token:
                cam.branch.set_focus(channel)
            if cam.seq != last:
                last = cam.seq
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                       + cam.snapshot() + b"\r\n")
            else:
                time.sleep(0.02)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.get("/shot/<int:eid>")
def shot(eid):
    with _events_lock:
        data = _shots.get(eid)
    if not data:
        return "yo'q", 404
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


if __name__ == "__main__":
    done, todo = detectors.status()
    for nm, br in nvr.BRANCHES.items():
        mark = "" if br.reachable else "  (ULANMADI)"
        print(f"Filial: {nm} · {len(br.cameras)} kamera{mark}")
    print(f"Qoidalar: {len(rules.load())} ta (Mars API)")
    print(f"Detektor: {len(done)} qoida qoplangan, {len(todo)} tasi hali yozilmagan")
    print(f"Yuz bazasi: {len(faces.known_faces())} xodim")
    print(f"http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
