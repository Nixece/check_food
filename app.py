import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ฟังก์ชันสำหรับดาวน์โหลดไฟล์ YOLO
def download_file(url, file_name):
    if not os.path.isfile(file_name):
        st.write(f"กำลังดาวน์โหลด {file_name} ...")
        urllib.request.urlretrieve(url, file_name)
        st.write(f"ดาวน์โหลด {file_name} เสร็จสิ้นแล้ว")

# URLs ของไฟล์ YOLO Tiny
yolo_weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# ดาวน์โหลดโมเดล YOLO Tiny
download_file(yolo_weights_url, "yolov3-tiny.weights")
download_file(yolo_cfg_url, "yolov3-tiny.cfg")
download_file(coco_names_url, "coco.names")

# โหลดโมเดล YOLO Tiny
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# โหลดคลาสจากไฟล์ coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ส่วนการอัปโหลดภาพ
st.title("ตรวจสอบบรรจุภัณฑ์อาหารจากมุมบน (Top View)")

uploaded_file = st.file_uploader("อัปโหลดภาพจากมุมบนของบรรจุภัณฑ์อาหาร (กล่องพลาสติก กล่องโฟม)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # แปลงภาพเป็น BGR สำหรับ OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ประมวลผลภาพด้วย YOLO Tiny
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # ดำเนินการประมวลผลผลลัพธ์
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # ตรวจจับเฉพาะประเภทที่อาจเป็นบรรจุภัณฑ์อาหาร เช่น กล่องพลาสติกหรือกล่องโฟม (cup, bowl)
            if confidence > 0.3 and classes[class_id] in ["cup", "bowl", "dining table"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # ตรวจสอบว่าเป็นวัตถุสี่เหลี่ยม (เช่นกล่อง)
                if 0.8 <= w / h <= 1.2:  # อัตราส่วนความกว้างและความสูงต้องใกล้เคียงกับสี่เหลี่ยมจัตุรัส
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # การกรองกล่องที่ทับซ้อนกัน
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        # วาดกรอบที่ตรวจพบและแสดงผล
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        st.write("ไม่พบวัตถุที่ตรงกับบรรจุภัณฑ์อาหารในภาพที่อัปโหลด")

    # แปลงกลับเป็น RGB และแสดงผลใน Streamlit
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="ผลลัพธ์หลังการตรวจสอบ", use_column_width=True)
