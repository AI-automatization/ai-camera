import av, numpy as np, math, time, sys, threading
CH = sys.argv[1] if len(sys.argv)>1 else "101"
url = f"rtsp://operator:0perator1audit@192.168.90.251:554/Streaming/Channels/{CH}"
print(f"kanal {CH} ga ulanyapti...", flush=True)
cont={"c":None,"stop":False}
def wd():
    time.sleep(30); cont["stop"]=True
    try: cont["c"] and cont["c"].close()
    except: pass
threading.Thread(target=wd,daemon=True).start()
try:
    c=av.open(url,options={"rtsp_transport":"tcp","timeout":"20000000","probesize":"500000"})
    cont["c"]=c
except Exception as e:
    print("Ulanmadi:",e); sys.exit(1)
astr=next((s for s in c.streams if s.type=="audio"),None)
if not astr:
    print("❌ audio track yo'q"); sys.exit(0)
print(f"✅ AUDIO: {astr.codec_context.name} {astr.codec_context.sample_rate}Hz — 6s tinglaymiz:",flush=True)
sr=astr.codec_context.sample_rate or 8000; win=int(sr*0.5); buf=[]; t0=time.time()
for frame in c.decode(audio=astr):
    buf.extend(frame.to_ndarray().flatten().astype(np.float64).tolist())
    while len(buf)>=win:
        x=np.array(buf[:win]); buf=buf[win:]
        rms=math.sqrt(float(np.mean(x**2))+1e-9); db=20*math.log10(rms/32768+1e-9)
        bars=int(max(0,min(40,(db+60)*40/60)))
        print(f"  {db:6.1f} dB  {'█'*bars}",flush=True)
    if cont["stop"] or time.time()-t0>6: break
c.close(); print("tayyor")
