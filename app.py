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
        # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

        # ใช้ Adaptive Threshold
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)

            # กรองพื้นที่ที่ถือว่าเป็นคราบ (เช่น ถ้าพื้นที่ใหญ่กว่า 300)
            if area > 300:  # ค่าที่ใช้กรองสำหรับคราบ
                continue  # ไม่นับเป็นเศษอาหาร

            # ตรวจสอบลักษณะอื่น ๆ เช่น ความยาวของ contour
            perimeter = cv2.arcLength(contour, True)
            if perimeter / area < 5:  # ถ้าความยาวมากเกินไปถือว่าเป็นคราบ
                continue

            # ถ้าไม่ใช่คราบ เพิ่มเข้าไปใน waste_pixels
            waste_pixels += area

        # คำนวณสัดส่วนของเศษอาหารที่เหลือ
        total_pixels = image_gray.size
        if total_pixels == 0:  # ตรวจสอบว่า total_pixels เป็น 0 หรือไม่
            return "ไม่สามารถประมวลผลภาพได้"

        waste_ratio = waste_pixels / total_pixels
        waste_percentage = waste_ratio * 100

        # แสดงผลลัพธ์
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

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result = check_food_waste_auto(image)
    st.write(result)
