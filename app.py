import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    # ปรับขนาดภาพเพื่อไม่ให้ใหญ่เกินไป
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารโดยใช้การระบุสีของบรรจุภัณฑ์
def check_food_waste_with_color(image, packaging_color):
    try:
        # แปลงภาพที่อัปโหลดเป็นรูปแบบ HSV (Hue, Saturation, Value) เพื่อให้ตรวจจับสีได้ง่ายขึ้น
        image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

        # กำหนดค่าช่วงสีของบรรจุภัณฑ์ที่ระบุ (สามารถปรับได้ตามความต้องการ)
        if packaging_color == 'ไม่ระบุ':
            lower_color = np.array([0, 0, 0])
            upper_color = np.array([180, 255, 255])  # ตรวจจับทุกสี
        elif packaging_color == 'ขาว':
            lower_color = np.array([0, 0, 200]) # สีขาวในรูปแบบ HSV
            upper_color = np.array([180, 20, 255])
        elif packaging_color == 'ดำ':
            lower_color = np.array([0, 0, 0]) # สีดำในรูปแบบ HSV
            upper_color = np.array([180, 255, 50])
        elif packaging_color == 'ใส':
            lower_color = np.array([0, 0, 240]) # สีใส (สว่างเกือบสุดใน HSV)
            upper_color = np.array([180, 20, 255])
        elif packaging_color == 'ฟ้า':
            lower_color = np.array([90, 50, 50]) # สีฟ้าในรูปแบบ HSV
            upper_color = np.array([130, 255, 255])
        elif packaging_color == 'เขียว':
            lower_color = np.array([35, 50, 50]) # สีเขียวในรูปแบบ HSV
            upper_color = np.array([85, 255, 255])
        elif packaging_color == 'แดง':
            lower_color = np.array([0, 50, 50]) # สีแดงในรูปแบบ HSV
            upper_color = np.array([10, 255, 255])

        # สร้างหน้ากากเพื่อแยกพื้นที่ที่ตรงกับสีของบรรจุภัณฑ์
        mask = cv2.inRange(image_hsv, lower_color, upper_color)

        # Invert mask เพื่อให้พื้นที่ที่ไม่ใช่บรรจุภัณฑ์เป็นพื้นที่ที่เราสนใจ (เศษอาหาร)
        mask_inv = cv2.bitwise_not(mask)

        # ใช้การค้นหา Contours เพื่อตรวจจับพื้นที่เศษอาหาร
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # คำนวณจำนวนพิกเซลของเศษอาหาร
        waste_pixels = sum(cv2.contourArea(contour) for contour in contours)

        # คำนวณจำนวนพิกเซลทั้งหมดในภาพ
        total_pixels = image_hsv.size

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

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์และการระบุสี
st.title('Food Waste Detection with Color Specification')

# ให้ผู้ใช้ระบุสีของบรรจุภัณฑ์
packaging_color = st.selectbox("กรุณาเลือกสีของบรรจุภัณฑ์", 
                               ['ไม่ระบุ', 'ขาว', 'ดำ', 'ใส', 'ฟ้า', 'เขียว', 'แดง'])

# อัปโหลดภาพบรรจุภัณฑ์
st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and packaging_color:
    image = Image.open(uploaded_file)
    
    # ปรับขนาดภาพ
    image = resize_image(image)

    # แสดงภาพที่อัปโหลด
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยใช้การระบุสีบรรจุภัณฑ์
    result = check_food_waste_with_color(image, packaging_color)
    st.write(result)
