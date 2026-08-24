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
from flask import (Flask, Response, jsonify, render_template_string,
                   request, send_file)

import nvr
import rules
import faces
import pose
import detectors

PORT = 5001
MAX_EVENTS = 60
ANALYZE_INTERVAL = 1.0     # bitta kamerani shu tezlikdan tez tahlil qilmaymiz

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


nvr.build(on_frame=draw)         # filiallarni yaratadi va ishga tushiradi
CAMERAS = nvr.all_cameras()      # {'Filial/kanal': Camera}
_last_analyzed = {}


def analyzer():
    """Bitta GPU — kameralarni navbat bilan tahlil qiladi, ochilgani birinchi.

    Barcha filiallar bitta navbatda: GPU bitta, ikkita analizator ip ochish
    faqat bir-birini kutishga olib keladi.
    """
    pose.model()          # oldindan yuklab qo'yamiz
    faces._models()
    order = list(CAMERAS)
    i = 0
    while True:
        cam = None
        for br in nvr.BRANCHES.values():        # ochilgan kamera navbatsiz
            ch = br.focused_channel()
            if ch and br.cameras[ch]._raw is not None:
                cam = br.cameras[ch]
                break
        if cam is None:
            cam = CAMERAS[order[i % len(order)]]
            i += 1

        if time.time() - _last_analyzed.get(cam.key, 0) < ANALYZE_INTERVAL:
            time.sleep(0.05)
            continue
        frame = cam.take_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        _last_analyzed[cam.key] = time.time()

        branch = cam.branch.name
        try:
            persons = pose.people(pose.infer(frame))
            found = faces.identify(frame, branch=branch)
        except Exception as e:
            print(f"[analyzer] {cam.key} tahlil xatosi: {e}")
            continue

        ctx = detectors.Context(branch=branch, channel=cam.channel,
                                camera_name=cam.name, faces=found, persons=persons)
        events = detectors.run(ctx)

        cam.apply({"faces": found, "persons": persons, "events": events,
                   "zone": ctx.zone, "count": ctx.head_count,
                   "named": ctx.named,
                   "identity": ctx.identity_reliable,
                   "face_px": max((f["box"][2] for f in found), default=0)})

        for ev in events:
            ev["camera"] = cam.name
            ev["channel"] = cam.channel
            ev["branch"] = branch
            vis = draw(frame.copy(), {"faces": found, "persons": persons,
                                      "events": [ev]})
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
            add_event(ev, buf.tobytes() if ok else b"")
            print(f"[hodisa] {branch}/{ev['rule_number']} ({ev['rule_type']}, "
                  f"{ev['score']} ball) — {ev['reason']}")


threading.Thread(target=analyzer, daemon=True).start()


PAGE = """
<!doctype html><meta charset=utf-8><title>MARS audit kamerasi</title>
<style>
 :root{--bg:#14110e;--card:#1e1a16;--line:#332c24;--fg:#e8e2d8;--dim:#9a9086}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
   gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:16px;margin:0;font-weight:600}
 .total{font-size:14px} .total b{font-size:22px;margin-right:4px}
 #tabs{display:flex;gap:6px}
 #tabs button{background:transparent;color:var(--dim);border:1px solid var(--line);
   border-radius:20px;padding:4px 14px;cursor:pointer;font-size:13px}
 #tabs button.act{background:var(--card);color:var(--fg);border-color:#6b5b45}
 .dim{color:var(--dim);font-size:13px}
 main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px;
   align-items:start}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
 .cam{background:var(--card);border:1px solid var(--line);border-radius:8px;
   overflow:hidden;cursor:pointer;transition:border-color .15s}
 .cam:hover{border-color:#6b5b45}
 .cam .shot{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;
   background:#0b0908}
 .cam .body{padding:8px 10px}
 .cam .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
 .cam .nm{font-weight:600;font-size:13px}
 .cam .zone{color:var(--dim);font-size:11px}
 .cam .cnt{display:flex;align-items:baseline;gap:6px;margin-top:4px}
 .cam .num{font-size:26px;font-weight:700;line-height:1}
 .cam .num.zero{color:var(--dim);font-weight:400}
 .cam .unit{color:var(--dim);font-size:12px}
 .cam .idbadge{margin-left:auto;font-size:10px;padding:1px 6px;border-radius:20px}
 .idok{background:#2d4a2d;color:#9fd89f} .idno{background:#3a3230;color:#b9a89a}
 .cam.off{opacity:.45}
 .cam.hit{outline:2px solid #d9534f}
 #big{position:fixed;inset:0;background:#000e;display:none;z-index:9;
   align-items:center;justify-content:center;flex-direction:column;gap:10px}
 #big.on{display:flex}
 #big img{max-width:94vw;max-height:82vh;border-radius:8px;background:#000}
 #bigbar{color:var(--fg);display:flex;gap:18px;align-items:center;font-size:14px}
 #bigbar button{background:var(--card);color:var(--fg);border:1px solid var(--line);
   border-radius:6px;padding:6px 14px;cursor:pointer;font-size:14px}
 aside{background:var(--card);border:1px solid var(--line);border-radius:8px;
   padding:12px;max-height:calc(100vh - 120px);overflow:auto}
 .ev{border-bottom:1px solid var(--line);padding:10px 0}
 .ev:last-child{border:0}
 .ev b{font-size:13px} .ev img{width:100%;border-radius:5px;margin-top:6px}
 .tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
   font-weight:600;margin-right:6px}
 .green{background:#2d4a2d;color:#9fd89f} .yellow{background:#4a432d;color:#e0d18a}
 .red{background:#4a2d2d;color:#efa0a0} .black{background:#3a3a3a;color:#ddd}
 .empty{color:var(--dim);padding:20px 0;text-align:center}
</style>
<header>
  <h1>MARS audit kamerasi</h1>
  <span id=tabs></span>
  <span class=total><b id=total>0</b> odam</span>
  <span class=dim id=meta>yuklanmoqda…</span>
</header>
<div id=big><img id=bigimg><div id=bigbar>
  <span id=bigname></span><span class=dim id=bigfps></span>
  <button onclick="closeBig()">Yopish</button></div></div>
<main>
  <div class=grid id=grid></div>
  <aside>
    <div style="font-weight:600;margin-bottom:8px">Hodisalar</div>
    <div id=events><div class=empty>hozircha yo'q</div></div>
  </aside>
</main>
<script>
const grid=document.getElementById('grid'), evbox=document.getElementById('events');
let built=false, branch=null;
function buildTabs(list){
  const box=document.getElementById('tabs');
  if(box.childElementCount===list.length) return;
  box.innerHTML='';
  for(const b of list){
    const t=document.createElement('button');
    t.textContent=b;
    t.onclick=()=>{ if(branch===b) return;
      branch=b; built=false; closeBig(); grid.innerHTML=''; tick(); };
    box.appendChild(t);
  }
}
function build(cams){
  grid.innerHTML='';
  for(const c of cams){
    const d=document.createElement('div'); d.className='cam'; d.id='c'+c.channel;
    d.innerHTML=`<img class=shot id="s${c.channel}">
      <div class=body>
        <div class=top><span class=nm>${c.name}</span>
          <span class=zone>${c.zone||''}</span></div>
        <div class=cnt><span class=num id="n${c.channel}">0</span>
          <span class=unit>odam</span>
          <span class="idbadge idno" id="b${c.channel}">—</span></div>
      </div>`;
    d.onclick=()=>openBig(c.channel,c.name);
    grid.appendChild(d);
  }
  built=true;
}
let bigCh=null;
function openBig(ch,name){
  bigCh=ch;
  document.getElementById('bigimg').src=
    '/stream/'+encodeURIComponent(branch)+'/'+ch+'?big=1';
  document.getElementById('bigname').textContent=name;
  document.getElementById('big').classList.add('on');
}
function closeBig(){
  bigCh=null;
  document.getElementById('bigimg').src='';   // oqimni uzamiz, fokus bo'shaydi
  document.getElementById('big').classList.remove('on');
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeBig();});
async function tick(){
  const s=await (await fetch('/state'+(branch?'?branch='+encodeURIComponent(branch):''))).json();
  branch=s.branch;
  buildTabs(s.branches);
  for(const t of document.getElementById('tabs').children)
    t.classList.toggle('act', t.textContent===branch);
  if(!built) build(s.cameras);
  const total=s.cameras.reduce((a,c)=>a+c.count,0);
  document.getElementById('total').textContent=total;
  const warn = s.locked ? `NVR QULFLANGAN — ${Math.ceil(s.lock_left/60)} daqiqa qoldi`
             : (!s.reachable ? 'NVR ga ulanmadi' : '');
  document.getElementById('meta').textContent =
    (warn ? warn+' · ' : '') +
    `${s.online}/${s.cameras.length} kamera · `+
    `${s.rules} qoida · ${s.detectors} detektor`;
  document.getElementById('meta').style.color = warn ? '#e08a8a' : '';
  for(const c of s.cameras){
    const el=document.getElementById('c'+c.channel);
    if(!el) continue;
    el.classList.toggle('hit', c.hit);
    el.classList.toggle('off', !c.online);
    const n=document.getElementById('n'+c.channel);
    n.textContent=c.count; n.classList.toggle('zero', c.count===0);
    const b=document.getElementById('b'+c.channel);
    b.textContent=c.identity?'yuz aniq':'yuz kichik';
    b.className='idbadge '+(c.identity?'idok':'idno');
    // Katta ko'rinish ochiq bo'lsa grid kadrlarini so'ramaymiz — butun
    // tezlik budjeti ochilgan kameraga ketsin.
    if(bigCh===null)
      document.getElementById('s'+c.channel).src=
        '/still/'+encodeURIComponent(branch)+'/'+c.channel+'?t='+Date.now();
    if(c.channel===bigCh)
      document.getElementById('bigfps').textContent=c.fps+' kadr/sek';
  }
  evbox.innerHTML = s.events.length ? s.events.map(e=>`
    <div class=ev>
      <span class="tag ${e.rule_type}">${e.rule_number} · ${e.score} ball</span>
      <b>${e.camera}</b>
      <div class=dim>${e.at.replace('T',' ')}</div>
      <div>${e.reason}</div>
      <div class=dim style="margin-top:4px">${e.rule_text}</div>
      <img src="/shot/${e.id}" loading=lazy>
    </div>`).join('') : "<div class=empty>hodisa yo'q</div>";
}
tick(); setInterval(tick,3000);
</script>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/state")
def state():
    """Bitta filial holati. ?branch=Nomi — qaysi filial (birinchisi standart)."""
    name = request.args.get("branch") or (nvr.ENABLED[0] if nvr.ENABLED else "")
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
            "fps": round(cam.fps, 1),
            "face_px": st.get("face_px", 0),
            "hit": cam.channel in hits,
        })
    done, _ = detectors.status()
    locked, left = br.lock_state() if br else (False, 0)
    return jsonify(branch=name, branches=list(nvr.BRANCHES),
                   reachable=(br.reachable is not False) if br else False,
                   cameras=cams, events=events,
                   online=sum(1 for c in cams if c["online"]),
                   locked=locked, lock_left=left,
                   rules=len(rules.load()), detectors=len(done))


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


@app.get("/stream/<branch>/<channel>")
def stream(branch, channel):
    cam = nvr.find(branch, channel)
    if cam is None:
        return "yo'q", 404
    big = request.args.get("big") == "1"

    def gen():
        last = -1
        while True:
            # Kimdir katta ko'rinishda qarab turibdi — fokusni ushlab turamiz.
            # Ilgari fokus faqat bosilganda yoqilardi va TTL 6 sekundda
            # so'nardi, ya'ni amalda hech qachon ishlamasdi: 15 kamera
            # birdek sekin so'ralib, hammasi 0.7 kadr/sek edi.
            if big:
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
