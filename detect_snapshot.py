from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # avtomatik yuklab oladi (~6MB)
img = cv2.imread("snapshot.jpg")
# class 0 = person; conf 0.3
res = model(img, classes=[0], conf=0.3, verbose=False)[0]
n = 0
for box in res.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"person {conf:.0%}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    n += 1
cv2.putText(img, f"Odamlar: {n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.imwrite("detected.jpg", img)
print(f"ANIQLANDI: {n} ta odam")
