import av, numpy as np, math, time, sys

CH = "1101"  # B4
url = f"rtsp://operator:0perator1audit@192.168.90.251:554/Streaming/Channels/{CH}"
print(f"B4 (kanal {CH}) audio oqimiga ulanyapti...")
try:
    container = av.open(url, options={"rtsp_transport":"tcp","stimeout":"8000000"})
except Exception as e:
    print("Ulanmadi:", e); sys.exit(1)

astream = next((s for s in container.streams if s.type=="audio"), None)
if astream is None:
    print("❌ Bu kamerada AUDIO track YO'Q (mikrofon yo'q yoki o'chirilgan)")
    sys.exit(1)
print(f"✅ Audio track topildi: {astream.codec_context.name}, "
      f"{astream.codec_context.sample_rate} Hz")

print("\n6 soniya tinglaymiz (ovoz darajasi dB):")
t0=time.time(); buf=[]; sr=astream.codec_context.sample_rate or 8000
win=int(sr*0.5)  # 0.5s oyna
def level(x):
    rms=math.sqrt(float(np.mean(x.astype(np.float64)**2))+1e-9)
    db=20*math.log10(rms/32768+1e-9)
    bars=int(max(0,min(40,(db+60)*40/60)))
    return db,"█"*bars
for frame in container.decode(audio=astream):
    arr=frame.to_ndarray().flatten()
    buf.extend(arr.tolist())
    while len(buf)>=win:
        chunk=np.array(buf[:win]); buf=buf[win:]
        db,bar=level(chunk)
        print(f"  {db:6.1f} dB  {bar}")
    if time.time()-t0>6: break
container.close()
print("\n(Baland ovoz => dB yuqori (0 ga yaqin), jim => -50/-60 dB)")
