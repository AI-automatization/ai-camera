import requests, time, math, audioop
from requests.auth import HTTPDigestAuth

IP="192.168.90.251"; USER="operator"; PW="0perator1audit"
s=requests.Session(); s.auth=HTTPDigestAuth(USER,PW)
base=f"http://{IP}/ISAPI/System/TwoWayAudio/channels"

for ch in ["1","2"]:
    print(f"\n=== Two-way audio kanal {ch} ===")
    # eski sessiyani yopamiz
    try: s.put(f"{base}/{ch}/close", timeout=5)
    except: pass
    time.sleep(0.5)
    # ochamiz
    try:
        r=s.put(f"{base}/{ch}/open", timeout=6)
        st=r.text
        if "OK" in st or r.status_code==200 and "Busy" not in st:
            print(f"  ochildi ✅")
        else:
            import re; m=re.search(r'<statusString>([^<]*)',st)
            print(f"  ochilmadi: {m.group(1) if m else st[:60]}")
            continue
    except Exception as e:
        print("  open xato:",e); continue
    # audioData o'qiymiz (~5s)
    try:
        r=s.get(f"{base}/{ch}/audioData", stream=True, timeout=8)
        raw=b""; t0=time.time()
        for chunk in r.iter_content(2048):
            raw+=chunk
            if time.time()-t0>5 or len(raw)>80000: break
        print(f"  audio keldi: {len(raw)} bayt")
        if len(raw)>2000:
            # G.711 ulaw -> PCM16, daraja
            try:
                pcm=audioop.ulaw2lin(raw,2)
                rms=audioop.rms(pcm,2)
                db=20*math.log10(rms/32768+1e-9)
                print(f"  🔊 OVOZ DARAJASI: {db:.1f} dB (rms={rms})")
            except Exception as e:
                print("  dekod xato:",e)
        else:
            print("  ovoz bo'sh/kelmadi")
    except Exception as e:
        print("  audioData xato:",e)
    finally:
        try: s.put(f"{base}/{ch}/close", timeout=5)
        except: pass
