import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันแปลงภาพเป็นรูปแบบสี HSV เพื่อใช้ในการตรวจจับสีที่คล้ายอาหาร
def is_food_basic(image):
    try:
        img_array = np.array(image)
        hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # กำหนดขอบเขตของสีที่พบบ่อยในอาหาร เช่น สีน้ำตาลและสีขาว
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_brown, mask_white)

        # คำนวณสัดส่วนพิกเซลที่ตรงกับสีอาหาร
        food_pixels = cv2.countNonZero(combined_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        food_ratio = food_pixels / total_pixels

        return food_ratio > 0.1  # ถ้าพิกเซลอาหารเกิน 10% ถือว่าเป็นภาพอาหาร
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")
        return False

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    # ปรับขนาดภาพเพื่อไม่ให้ใหญ่เกินไป
    return ImageOps.contain(image, max_size)

# ฟังก์ชันการคำนวณเศษอาหารที่เหลือในบรรจุภัณฑ์
def check_food_waste(image):
    try:
        gray_image = image.convert("L")  # แปลงภาพเป็นขาวดำ
        img_array = np.array(gray_image)

        # ใช้การทำ Threshold เพื่อแยกเศษอาหารออกจากพื้นหลัง
        _, threshold_image = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
        
        # ค้นหา Contours
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # คำนวณจำนวนพิกเซลของเศษอาหาร (ใช้การหาพื้นที่ Contours)
        waste_pixels = sum(cv2.contourArea(contour) for contour in contours)
        
        # ถ้ามีเศษอาหารเหลือมากกว่า 100 พิกเซล ถือว่ายังมีเศษอาหาร
        if waste_pixels > 100:
            return "ยังเหลืออาหารอยู่"
        else:
            return "บรรจุภัณฑ์ไม่เหลืออาหารเลย"
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด"

# ส่วนของการอัปโหลดรูปภาพ
st.title('Food Waste LE02')
st.write("""
    🚨 **คำแนะนำ**: กรุณาอัปโหลดภาพบรรจุภัณฑ์อาหาร เพื่อประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่
""")

# ส่วนการอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกภาพที่ต้องการอัปโหลด", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # ปรับขนาดภาพ
    image = resize_image(image)

    st.image(image, caption="ภาพที่คุณอัปโหลด", use_column_width=True)
    
    # ตรวจสอบว่าเป็นอาหารหรือไม่
    if not is_food_basic(image):
        st.write("กรุณาอัปโหลดภาพที่เป็นบรรจุภัณฑ์อาหารเท่านั้น")
    else:
        # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่
        result = check_food_waste(image)
        st.write(result)
