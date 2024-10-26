import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการตรวจจับขนาดบรรจุภัณฑ์โดยอัตโนมัติ
def detect_package_size(image):
    # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
    image_array = np.array(image)
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # ใช้ Gaussian Blur เพื่อลด noise
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # ตรวจจับขอบของภาพโดยใช้ Canny Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # ค้นหา Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # หาพื้นที่ของ Contour ที่ใหญ่ที่สุด (ซึ่งน่าจะเป็นบรรจุภัณฑ์)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area

    # คืนค่าพื้นที่ของบรรจุภัณฑ์ (หน่วยเป็นพิกเซล)
    return max_area

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารและขนาดบรรจุภัณฑ์
def check_food_waste_auto_with_package(image):
    try:
        # ตรวจจับขนาดบรรจุภัณฑ์โดยอัตโนมัติ
        package_area = detect_package_size(image)

        # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(image_gray, (3, 3), 0)  # ปรับ kernel ขนาด 3x3

        # ใช้ Adaptive Threshold เพื่อตรวจจับเศษอาหาร
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours ของเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # ปรับเกณฑ์การกรองเป็น 50 พิกเซลแทนที่จะเป็น 100
                continue
            waste_pixels += area

        # แสดง Contour ที่ตรวจจับได้เพื่อตรวจสอบ
        contour_image = cv2.drawContours(image_array.copy(), contours, -1, (0, 255, 0), 2)
        st.image(contour_image, caption="Contour ที่ตรวจพบ", use_column_width=True)

        # คำนวณสัดส่วนของเศษอาหารเมื่อเทียบกับขนาดบรรจุภัณฑ์ที่ตรวจจับได้
        if package_area > 0:
            waste_ratio = waste_pixels / package_area
            waste_percentage = waste_ratio * 100
        else:
            waste_percentage = 100  # หากไม่พบขนาดบรรจุภัณฑ์ ให้ตั้งค่าเป็น 100%

        # แสดงผลลัพธ์
        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", False

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = resize_image(image)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติพร้อมตรวจจับขนาดบรรจุภัณฑ์
    result, passed = check_food_waste_auto_with_package(image)
    st.write(result)
