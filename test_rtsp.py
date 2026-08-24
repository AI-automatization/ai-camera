import cv2, sys, os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
url = "rtsp://operator:0perator1audit@192.168.90.251:554/Streaming/Channels/101"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("ULANMADI"); sys.exit(1)
ok = False
for i in range(30):
    ret, frame = cap.read()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        cv2.imwrite("snapshot.jpg", frame)
        print(f"KADR OLINDI: {w}x{h}, snapshot.jpg saqlandi")
        ok = True
        break
cap.release()
if not ok:
    print("Oqim ochildi lekin kadr kelmadi")
