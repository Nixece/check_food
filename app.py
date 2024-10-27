import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ส่วนการอัปโหลดภาพ
st.title("ตรวจจับบรรจุภัณฑ์จากรูปทรงสี่เหลี่ยม (ขอบโค้งมนหรือเหลี่ยม)")

uploaded_file = st.file_uploader("อัปโหลดภาพที่ต้องการตรวจสอบ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # โหลดภาพจากการอัปโหลดและแปลงเป็น RGB
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # แปลงภาพเป็น Grayscale และใช้ GaussianBlur เพื่อลด noise
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # ตรวจจับขอบในภาพด้วย Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # ค้นหา Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # วาดกรอบรอบ Contours ที่มีลักษณะใกล้เคียงสี่เหลี่ยม (ขอบมนหรือเหลี่ยม)
    for contour in contours:
        # ใช้ bounding box ที่มีขอบมนรองรับการเบี้ยวเพื่อครอบ contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # แปลงเป็นจำนวนเต็ม

        # คำนวณอัตราส่วนความกว้างต่อความสูงของ bounding box
        w, h = rect[1]
        if w > 0 and h > 0:
            aspect_ratio = max(w, h) / min(w, h)

            # กรองเฉพาะวัตถุที่มีอัตราส่วนใกล้เคียงกับสี่เหลี่ยมและมีขนาดใหญ่พอสมควร
            if 0.5 <= aspect_ratio <= 2.0 and max(w, h) > 50:
                # วาดกรอบสี่เหลี่ยมที่ขอบมนหรือเหลี่ยมรอบ contour ที่ตรวจพบ
                cv2.drawContours(image_bgr, [box], 0, (0, 255, 0), 2)

    # แปลงกลับเป็น RGB เพื่อแสดงใน Streamlit
    image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_result, caption="ผลลัพธ์หลังการตรวจจับบรรจุภัณฑ์ที่มีลักษณะขอบโค้งมนหรือเหลี่ยม", use_column_width=True)
