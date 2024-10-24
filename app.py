import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารและคราบ
def check_food_waste_auto(image):
    try:
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # ปรับ blockSize และ C
        threshold_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 9, 2)  # ปรับค่าใหม่

        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # ปรับค่าพื้นที่ที่ใช้กรองให้เล็กลง
            if area > 150:  # ลดเกณฑ์เหลือ 150
                continue
            waste_pixels += area

        total_pixels = image_gray.size
        waste_ratio = waste_pixels / total_pixels
        waste_percentage = waste_ratio * 100

        if waste_ratio < 0.04:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)"
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)"
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด"

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = resize_image(image)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)
    result = check_food_waste_auto(image)
    st.write(result)
