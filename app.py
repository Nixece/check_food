import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    # ปรับขนาดภาพเพื่อไม่ให้ใหญ่เกินไป
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารแบบอัตโนมัติ
def check_food_waste_auto(image):
    try:
        # แปลงภาพที่อัปโหลดเป็นขาวดำ
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # ใช้ Adaptive Threshold เพื่อปรับการตรวจจับ
        threshold_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours เพื่อตรวจจับเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # คำนวณจำนวนพิกเซลของเศษอาหาร (ใช้การหาพื้นที่ Contours)
        waste_pixels = sum(cv2.contourArea(contour) for contour in contours)

        # คำนวณจำนวนพิกเซลทั้งหมดในภาพ
        total_pixels = image_gray.size

        # คำนวณสัดส่วนของเศษอาหารที่เหลือ
        waste_ratio = waste_pixels / total_pixels

        # ถ้าเศษอาหารเหลือน้อยกว่า 2% ถือว่าไม่เหลืออาหาร
        if waste_ratio < 0.02:
            return "บรรจุภัณฑ์ไม่เหลืออาหารเลย"
        else:
            return "ยังเหลืออาหารอยู่"
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด"

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

# อัปโหลดภาพบรรจุภัณฑ์
st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # ปรับขนาดภาพ
    image = resize_image(image)

    # แสดงภาพที่อัปโหลด
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result = check_food_waste_auto(image)
    st.write(result)
