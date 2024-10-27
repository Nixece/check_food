import os
import urllib.request
import cv2
import numpy as np

# ฟังก์ชันสำหรับดาวน์โหลดไฟล์ถ้ายังไม่มีอยู่
def download_file(url, file_name):
    if not os.path.isfile(file_name):
        print(f"กำลังดาวน์โหลด {file_name} ...")
        urllib.request.urlretrieve(url, file_name)
        print(f"ดาวน์โหลด {file_name} เสร็จสิ้นแล้ว")

# URLs ของไฟล์ YOLO ที่ต้องการ
yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# ดาวน์โหลดไฟล์ YOLO
download_file(yolo_weights_url, "yolov3.weights")
download_file(yolo_cfg_url, "yolov3.cfg")
download_file(coco_names_url, "coco.names")

# โหลด YOLO ที่ฝึกมาแล้ว
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# โหลดคลาสจากไฟล์ coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# โหลดภาพที่ต้องการตรวจสอบ
image = cv2.imread("image.jpg")  # เปลี่ยนชื่อไฟล์ตามที่ต้องการ
height, width, channels = image.shape

# เตรียมภาพสำหรับ YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# ประมวลผลผลลัพธ์
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # กรองวัตถุตามความมั่นใจ
        if confidence > 0.5:
            if classes[class_id] == "bottle" or classes[class_id] == "box":  # ตัวอย่างสำหรับบรรจุภัณฑ์
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# การกรองกล่องที่ทับซ้อนกัน
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# วาดกรอบที่ตรวจพบและแสดงผล
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)  # สีของกรอบ

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# แสดงภาพ
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
