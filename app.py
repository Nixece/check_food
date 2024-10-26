import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับคำนวณตัวปรับคูณตามเปอร์เซ็นต์
def dynamic_scaling_factor(percentage):
    if percentage < 50:
        # ถ้าเปอร์เซ็นต์น้อยกว่า 50 จะคูณตามสัดส่วนจาก 1 ไปจนถึง 2
        return 1 + (percentage / 50)  # ตัวคูณจะเพิ่มจาก 1 ไปถึง 2 ที่ 50%
    else:
        # ถ้าเปอร์เซ็นต์มากกว่า 50 จะคูณน้อยลงเรื่อยๆ จาก 2 ไปถึง 1
        return 2 - ((percentage - 50) / 50)  # ตัวคูณจะลดจาก 2 ไปถึง 1 ที่ 100%

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

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารและปรับสัดส่วนอาหารที่เหลือด้วย dynamic scaling factor
def check_food_waste_auto_with_dynamic_scaling(image):
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

        # แสดง Contour ที่ตรวจจับได้เพื่อตรวจสอบ
        contour_image = cv2.drawContours(image_array.copy(), contours, -1, (0, 255, 0), 2)
        st.image(contour_image, caption="Contour ที่ตรวจพบ", use_column_width=True)

        # คำนวณเปอร์เซ็นต์ของอาหารที่เหลือจากพื้นที่เศษอาหาร
        if package_area > 0:
            food_remaining_percentage = (waste_pixels / package_area) * 100
        else:
            food_remaining_percentage = 100  # หากไม่พบขนาดบรรจุภัณฑ์ ให้ตั้งค่าเป็น 100%

        # คำนวณตัวคูณตามเปอร์เซ็นต์
        scaling_factor = dynamic_scaling_factor(food_remaining_percentage)

        # ปรับสัดส่วนอาหารที่เหลือโดยใช้ตัวคูณ
        food_remaining_percentage_adjusted = min(food_remaining_percentage * scaling_factor, 100)

        # แสดงผลลัพธ์
        return f"เปอร์เซ็นต์อาหารที่เหลือ (ปรับแล้ว): {food_remaining_percentage_adjusted:.2f}%", food_remaining_percentage_adjusted

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", 0

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = resize_image(image)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่าอาหารเหลืออยู่เท่าไหร่และปรับค่าให้เหมาะสมด้วย dynamic scaling factor
    result, food_remaining_percentage_adjusted = check_food_waste_auto_with_dynamic_scaling(image)
    st.write(result)
