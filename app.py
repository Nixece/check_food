import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    # ปรับขนาดภาพเพื่อไม่ให้ใหญ่เกินไป
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการลบพื้นหลังออกและตรวจจับเศษอาหาร
def check_food_waste_with_bg_subtraction(image, background):
    try:
        # แปลงภาพพื้นหลังและภาพที่อัปโหลดเป็นขาวดำ
        background_gray = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2GRAY)
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # ใช้การลบพื้นหลัง (Background Subtraction)
        diff_image = cv2.absdiff(background_gray, image_gray)

        # ใช้ Threshold เพื่อตรวจจับเศษอาหาร
        _, threshold_image = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)

        # ค้นหา Contours เพื่อตรวจจับเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # คำนวณจำนวนพิกเซลของเศษอาหาร (ใช้การหาพื้นที่ Contours)
        waste_pixels = sum(cv2.contourArea(contour) for contour in contours)

        # คำนวณจำนวนพิกเซลทั้งหมดในภาพ
        total_pixels = image_gray.size

        # คำนวณสัดส่วนของเศษอาหารที่เหลือ
        waste_ratio = waste_pixels / total_pixels

        # ถ้าเศษอาหารเหลือน้อยกว่า 5% ถือว่าไม่เหลืออาหาร
        if waste_ratio < 0.05:
            return "บรรจุภัณฑ์ไม่เหลืออาหารเลย"
        else:
            return "ยังเหลืออาหารอยู่"
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด"

# ส่วนของการอัปโหลดภาพพื้นหลังและภาพบรรจุภัณฑ์
st.title('Food Waste Detection with Background Subtraction')

# อัปโหลดภาพพื้นหลัง
st.write("กรุณาอัปโหลดภาพพื้นหลังที่ไม่มีบรรจุภัณฑ์")
background_file = st.file_uploader("เลือกภาพพื้นหลัง", type=["jpg", "png", "jpeg"])

# อัปโหลดภาพบรรจุภัณฑ์
st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if background_file is not None and uploaded_file is not None:
    background = Image.open(background_file)
    image = Image.open(uploaded_file)
    
    # ปรับขนาดภาพ
    background = resize_image(background)
    image = resize_image(image)

    # แสดงภาพที่อัปโหลด
    st.image(background, caption="ภาพพื้นหลัง", use_column_width=True)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่
    result = check_food_waste_with_bg_subtraction(image, background)
    st.write(result)
