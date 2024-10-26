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

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารและคำนวณเปอร์เซ็นต์
def check_food_waste_percentage(image):
    try:
        # ตรวจจับขนาดบรรจุภัณฑ์โดยอัตโนมัติ
        package_area = detect_package_size(image)

        # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(image_gray, (3, 3), 0)

        # ใช้ Adaptive Threshold เพื่อตรวจจับเศษอาหาร
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours ของเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # กรอง Contour เล็กๆ
                continue
            waste_pixels += area

        # คำนวณเปอร์เซ็นต์ของพื้นที่เศษอาหารเทียบกับบรรจุภัณฑ์
        if package_area > 0:
            food_remaining_percentage = (waste_pixels / package_area) * 100
        else:
            food_remaining_percentage = 100  # หากไม่พบขนาดบรรจุภัณฑ์ ให้ตั้งค่าเป็น 100%

        # แสดงผลลัพธ์เป็นเปอร์เซ็นต์ของพื้นที่เศษอาหาร
        return f"เปอร์เซ็นต์พื้นที่เศษอาหารที่เหลือ: {food_remaining_percentage:.2f}%"

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

    # ประเมินว่าอาหารเหลืออยู่เท่าไหร่โดยใช้เปอร์เซ็นต์ของพื้นที่เศษอาหาร
    result = check_food_waste_percentage(image)
    st.write(result)
