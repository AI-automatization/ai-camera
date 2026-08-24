import av, sys, threading

CH = sys.argv[1] if len(sys.argv)>1 else "1101"
url = f"rtsp://operator:0perator1audit@192.168.90.251:554/Streaming/Channels/{CH}"
cont = {"c":None}
def watchdog():
    import time; time.sleep(12)
    try:
        if cont["c"]: cont["c"].close()
    except: pass
threading.Thread(target=watchdog, daemon=True).start()
try:
    c = av.open(url, options={"rtsp_transport":"tcp","timeout":"5000000"})
    cont["c"]=c
    astr=[s for s in c.streams if s.type=="audio"]
    vstr=[s for s in c.streams if s.type=="video"]
    print(f"kanal {CH}: video={len(vstr)}, audio={len(astr)}")
    if astr:
        print(f"  ✅ AUDIO BOR: {astr[0].codec_context.name} {astr[0].codec_context.sample_rate}Hz")
    else:
        print("  ❌ audio track yo'q")
    c.close()
except Exception as e:
    print(f"kanal {CH}: xato — {e}")
