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

# ฟังก์ชันหาความแตกต่างระหว่างพื้นหลังและบรรจุภัณฑ์
def detect_package(background, package_image):
    # แปลงภาพเป็นขาวดำ
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    package_gray = cv2.cvtColor(package_image, cv2.COLOR_RGB2GRAY)
    
    # หาความแตกต่างระหว่างภาพพื้นหลังและภาพบรรจุภัณฑ์
    diff = cv2.absdiff(background_gray, package_gray)
    
    # ตั้งค่า Threshold เพื่อตรวจจับเฉพาะพื้นที่ที่แตกต่างกัน (บรรจุภัณฑ์)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # กรอง Contours เล็ก ๆ ออก
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # เลือกค่า 500 เพื่อลบพื้นที่ขนาดเล็ก
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    # ใช้ Mask ที่ปรับปรุงในการลบพื้นหลัง
    package_detected = cv2.bitwise_and(package_image, package_image, mask=mask)
    
    # คำนวณพื้นที่บรรจุภัณฑ์จากภาพที่เหลืออยู่
    total_pixels = cv2.countNonZero(cv2.cvtColor(package_detected, cv2.COLOR_RGB2GRAY))
    
    return package_detected, total_pixels

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารในบรรจุภัณฑ์
def check_food_waste_auto(image, total_pixels):
    try:
        # แปลงเป็นภาพขาวดำ
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(image_gray, (7, 7), 0)

        # ใช้ Threshold แบบคงที่
        _, threshold_image = cv2.threshold(blurred_image, 120, 255, cv2.THRESH_BINARY_INV)

        # ค้นหา Contours ของเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # กรองเฉพาะ Contours ที่ใหญ่พอ
                continue
            waste_pixels += area

        waste_ratio = waste_pixels / total_pixels
        waste_percentage = waste_ratio * 100

        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", False

# ฟังก์ชันสำหรับการสร้าง QR Code
def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    return byte_im

# ส่วนของการอัปโหลดภาพพื้นหลังและบรรจุภัณฑ์
st.title('Food Waste Detection (Automatic)')

st.write("กรุณาอัปโหลดภาพพื้นหลัง (ถ้ามี)")
background_file = st.file_uploader("เลือกภาพพื้นหลัง", type=["jpg", "png", "jpeg"], key="background")

st.write("กรุณาอัปโหลดภาพที่มีบรรจุภัณฑ์")
package_file = st.file_uploader("เลือกภาพที่มีบรรจุภัณฑ์", type=["jpg", "png", "jpeg"], key="package")

if package_file is not None:
    package_image = Image.open(package_file)
    package_image = resize_image(package_image)
    package_array = np.array(package_image)
    
    # หากมีภาพพื้นหลัง ให้ใช้การตรวจจับความแตกต่าง
    if background_file is not None:
        background_image = Image.open(background_file)
        background_image = resize_image(background_image)
        background_array = np.array(background_image)
        
        # ตรวจจับบรรจุภัณฑ์โดยหาความแตกต่างและคำนวณพื้นที่บรรจุภัณฑ์
        package_detected, total_pixels = detect_package(background_array, package_array)
        st.image(package_detected, caption="บรรจุภัณฑ์ที่ตรวจจับได้", use_column_width=True)
    else:
        # หากไม่มีพื้นหลัง ใช้ภาพทั้งหมดในการตรวจจับเศษอาหารและคำนวณพิกเซลในภาพทั้งหมด
        package_detected = package_array
        total_pixels = cv2.countNonZero(cv2.cvtColor(package_detected, cv2.COLOR_RGB2GRAY))
        st.image(package_detected, caption="ภาพที่มีบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result, passed = check_food_waste_auto(package_detected, total_pixels)
    st.write(result)

    # หากผ่านการตรวจสอบว่าไม่เหลืออาหาร
    if passed:
        st.success("บรรจุภัณฑ์นี้ไม่เหลือเศษอาหาร รับ 10 คะแนน!")
        
        # สร้าง QR Code ที่มีข้อมูลเฉพาะ
        qr_code_image = generate_qr_code("รหัสบรรจุภัณฑ์นี้สำหรับสะสม 10 คะแนน")
        
        # แสดง QR Code
        st.image(qr_code_image, caption="QR Code สำหรับสะสมแต้ม", use_column_width=False)
