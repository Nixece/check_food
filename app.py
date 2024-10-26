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

# ฟังก์ชันสำหรับการตรวจจับบรรจุภัณฑ์จากขอบภาพเข้ามากลาง
def detect_packaging_from_edges(image):
    # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
    image_array = np.array(image)
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # ใช้ Gaussian Blur เพื่อลด noise
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # ใช้ Canny edge detection เพื่อตรวจจับขอบของบรรจุภัณฑ์
    edges = cv2.Canny(blurred_image, 100, 200)

    # ใช้การไล่จากขอบของภาพ (สร้างกรอบ)
    height, width = edges.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # สร้างกรอบขอบภาพที่ใหญ่ขึ้นจากขอบเข้ามาด้านใน (4 ด้าน)
    cv2.rectangle(mask, (10, 10), (width - 10, height - 10), 255, -1)  # ทำกรอบรอบขอบภาพ
    edges = cv2.bitwise_and(edges, mask)

    # ค้นหา Contours สำหรับบรรจุภัณฑ์
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ตรวจจับ Contour ที่ใหญ่ที่สุดเพื่อเป็นบรรจุภัณฑ์
    largest_contour = max(contours, key=cv2.contourArea)

    # สำเนาภาพสำหรับวาดขอบเขตบรรจุภัณฑ์ที่ใหญ่ที่สุด
    packaging_detected = image_array.copy()
    cv2.drawContours(packaging_detected, [largest_contour], -1, (0, 255, 0), 2)  # วาดขอบเขตบรรจุภัณฑ์ที่ใหญ่ที่สุด

    # สร้าง mask สำหรับพื้นที่บรรจุภัณฑ์ที่ตรวจพบ
    mask = np.zeros_like(image_gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return packaging_detected, mask, largest_contour

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารภายในบรรจุภัณฑ์
def check_food_waste_auto(image, mask):
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

        # ค้นหา Contours สำหรับเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # สำเนาภาพสำหรับวาดขอบเขตเศษอาหาร
        food_waste_detected = image_array.copy()

        waste_pixels = 0
        packaging_area = cv2.countNonZero(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            # ตรวจสอบว่าเศษอาหารอยู่ภายในพื้นที่บรรจุภัณฑ์ (ใช้ mask)
            if area < 100 or cv2.pointPolygonTest(mask, tuple(contour[0][0]), False) < 0:
                continue  # ข้ามเศษอาหารที่อยู่นอกพื้นที่บรรจุภัณฑ์
            waste_pixels += area  # เก็บพื้นที่ของเศษอาหาร
            cv2.drawContours(food_waste_detected, [contour], -1, (0, 0, 255), 2)  # วาดขอบเขตเศษอาหาร

        # คำนวณสัดส่วนของเศษอาหารที่เหลือเทียบกับพื้นที่บรรจุภัณฑ์
        waste_ratio = waste_pixels / packaging_area if packaging_area > 0 else 0
        waste_percentage = waste_ratio * 100

        # แสดงผลลัพธ์
        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True, food_waste_detected
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False, food_waste_detected
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", False, None

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

    # ตรวจจับบรรจุภัณฑ์จากขอบภาพเข้ามาตรงกลาง
    packaging_img, mask, largest_contour = detect_packaging_from_edges(image)
    st.image(packaging_img, caption="พื้นที่บรรจุภัณฑ์ที่ตรวจจับได้", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result, passed, waste_img = check_food_waste_auto(image, mask)
    st.write(result)

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
