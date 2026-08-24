import cv2, os, sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="rtsp_transport;tcp"
U,P,H="operator","0perator1audit","192.168.90.251"
chans=["101","201","301","401","601","901","1001","1101","1201","1301","1401","1501","1601","1701","1801"]
for ch in chans:
    url=f"rtsp://{U}:{P}@{H}:554/Streaming/Channels/{ch}"
    cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG)
    ok=False; wh=""
    if cap.isOpened():
        for _ in range(15):
            r,f=cap.read()
            if r and f is not None:
                ok=True; wh=f"{f.shape[1]}x{f.shape[0]}"; break
    cap.release()
    print(f"{ch}: {'OK '+wh if ok else 'YOQ'}", flush=True)
