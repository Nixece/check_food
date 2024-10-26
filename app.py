import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps
import qrcode
import io

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(500, 500)):
    return ImageOps.contain(image, max_size)

# ฟังก์ชันสำหรับการตรวจจับพื้นที่บรรจุภัณฑ์ที่ใหญ่ที่สุดและเศษอาหาร
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

        # ค้นหา Contours ทั้งหมด
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ตรวจจับ Contour ที่ใหญ่ที่สุดเพื่อเป็นบรรจุภัณฑ์
        largest_contour = max(contours, key=cv2.contourArea)

        # สำเนาภาพสำหรับวาดขอบเขตบรรจุภัณฑ์ที่ใหญ่ที่สุด
        packaging_detected = image_array.copy()
        cv2.drawContours(packaging_detected, [largest_contour], -1, (0, 255, 0), 2)  # วาดขอบเขตบรรจุภัณฑ์ที่ใหญ่ที่สุด

        # หาพื้นที่บรรจุภัณฑ์ที่ใหญ่ที่สุด
        packaging_area = cv2.contourArea(largest_contour)

        # สำเนาภาพสำหรับวาดขอบเขตเศษอาหาร
        food_waste_detected = image_array.copy()

        # ค้นหา Contours สำหรับเศษอาหาร
        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # ปรับเกณฑ์พื้นที่ที่ใช้กรองให้เหมาะสม (นับเฉพาะเศษอาหารที่ไม่ใช่บรรจุภัณฑ์ที่ใหญ่ที่สุด)
            if area < 100 or contour is largest_contour:  # ข้ามบรรจุภัณฑ์ที่ใหญ่ที่สุด
                continue
            waste_pixels += area  # เก็บพื้นที่ของเศษอาหาร
            cv2.drawContours(food_waste_detected, [contour], -1, (0, 0, 255), 2)  # วาดขอบเขตเศษอาหาร

        # คำนวณสัดส่วนของเศษอาหารที่เหลือเทียบกับบรรจุภัณฑ์
        waste_ratio = waste_pixels / packaging_area if packaging_area > 0 else 0
        waste_percentage = waste_ratio * 100

        # แสดงผลลัพธ์
        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True, packaging_detected, food_waste_detected
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False, packaging_detected, food_waste_detected
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", False, None, None

# ฟังก์ชันสำหรับการสร้าง QR Code
def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    
    # แปลงภาพ QR Code เป็นฟอร์แมตที่ Streamlit รองรับ
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    return byte_im

# ส่วนของการอัปโหลดภาพบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

st.write("กรุณาอัปโหลดภาพบรรจุภัณฑ์ที่ต้องการตรวจสอบเศษอาหาร")
uploaded_file = st.file_uploader("เลือกภาพบรรจุภัณฑ์", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = resize_image(image)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result, passed, packaging_img, waste_img = check_food_waste_auto(image)
    st.write(result)

    # แสดงภาพบรรจุภัณฑ์ที่ตรวจจับได้
    if packaging_img is not None:
        st.image(packaging_img, caption="พื้นที่บรรจุภัณฑ์ที่ตรวจจับได้", use_column_width=True)

    # แสดงภาพเศษอาหารที่ตรวจจับได้
    if waste_img is not None:
        st.image(waste_img, caption="พื้นที่เศษอาหารที่ตรวจจับได้", use_column_width=True)

    # หากผ่านการตรวจสอบว่าไม่เหลืออาหาร
    if passed:
        st.success("บรรจุภัณฑ์นี้ไม่เหลือเศษอาหาร รับ 10 คะแนน!")
        
        # สร้าง QR Code ที่มีข้อมูลเฉพาะ
        qr_code_image = generate_qr_code("รหัสบรรจุภัณฑ์นี้สำหรับสะสม 10 คะแนน")
        
        # แสดง QR Code
        st.image(qr_code_image, caption="QR Code สำหรับสะสมแต้ม", use_column_width=False)
