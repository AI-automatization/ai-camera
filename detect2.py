from ultralytics import YOLO
import cv2
model = YOLO("yolov8m.pt")
img = cv2.imread("snapshot.jpg")
res = model(img, classes=[0], conf=0.20, imgsz=1280, verbose=False)[0]
n=0
for box in res.boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0]); conf=float(box.conf[0])
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(img,f"{conf:.0%}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    n+=1
cv2.putText(img,f"Odamlar: {n}",(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3)
cv2.imwrite("detected.jpg", img)
print(f"ANIQLANDI: {n} ta odam")
