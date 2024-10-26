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

# ฟังก์ชันสำหรับการตรวจจับบรรจุภัณฑ์
def detect_packaging(image):
    image_array = np.array(image)
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # ใช้ Gaussian Blur เพื่อลด noise
    blurred_image = cv2.GaussianBlur(image_gray, (7, 7), 0)

    # ใช้ Canny edge detection เพื่อตรวจจับขอบของบรรจุภัณฑ์
    edges = cv2.Canny(blurred_image, 50, 150)

    # ใช้การแปลง Morphological เพื่อลบ noise เพิ่มเติม
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ค้นหา Contours สำหรับบรรจุภัณฑ์
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ตรวจจับ Contour ที่ใหญ่ที่สุดที่มีรูปร่างเป็นสี่เหลี่ยมหรือวงกลม
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            # ตรวจสอบว่า Contour มีรูปร่างเป็นสี่เหลี่ยมหรือวงกลม
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # ถ้ามี 4 จุดแสดงว่าเป็นสี่เหลี่ยม
                largest_contour = contour
                max_area = area
            else:
                # ตรวจสอบความกลม (circularity) สำหรับวงกลม
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter ** 2))
                    if 0.7 < circularity < 1.3:  # ถ้าค่าความกลมใกล้เคียง 1 ถือว่าเป็นวงกลม
                        largest_contour = contour
                        max_area = area

    # ตรวจจับพื้นที่บรรจุภัณฑ์
    if largest_contour is not None:
        mask = np.zeros_like(image_gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # วาดขอบเขตบรรจุภัณฑ์บนภาพ
        packaging_detected = image_array.copy()
        cv2.drawContours(packaging_detected, [largest_contour], -1, (0, 255, 0), 2)

        return mask, packaging_detected
    else:
        return None, None

# ฟังก์ชันสำหรับการตรวจจับเศษอาหารในบรรจุภัณฑ์
def check_food_waste_auto(image, packaging_mask):
    try:
        # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ใช้ Mask ของบรรจุภัณฑ์เพื่อให้เหลือเฉพาะพื้นที่บรรจุภัณฑ์
        masked_image = cv2.bitwise_and(image_gray, image_gray, mask=packaging_mask)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 0)

        # ใช้ Adaptive Threshold เพื่อตรวจจับเศษอาหาร
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours สำหรับเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        waste_detected = image_array.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # กำหนดพื้นที่ขั้นต่ำเพื่อตัด Noise ออก
                continue

            # ตรวจสอบว่าเศษอาหารอยู่ในพื้นที่บรรจุภัณฑ์โดยดูจาก packaging_mask
            for point in contour:
                x, y = point[0]
                if packaging_mask[y, x] == 255:  # ตรวจสอบว่าอยู่ในพื้นที่บรรจุภัณฑ์หรือไม่
                    waste_pixels += area
                    cv2.drawContours(waste_detected, [contour], -1, (0, 0, 255), 2)
                    break  # ออกจากลูปเมื่อพบว่า contour นี้เป็นเศษอาหารที่อยู่ในบรรจุภัณฑ์

        # คำนวณสัดส่วนของเศษอาหารที่เหลือ
        total_pixels = cv2.countNonZero(packaging_mask)
        waste_ratio = waste_pixels / total_pixels if total_pixels > 0 else 0
        waste_percentage = min(waste_ratio * 100, 100)  # จำกัดไม่ให้เกิน 100%

        # แสดงผลลัพธ์
        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True, waste_detected
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False, waste_detected
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

    # ตรวจจับบรรจุภัณฑ์
    packaging_mask, packaging_img = detect_packaging(image)
    
    if packaging_mask is not None:
        st.image(packaging_img, caption="พื้นที่บรรจุภัณฑ์ที่ตรวจจับได้", use_column_width=True)

        # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
        result, passed, waste_img = check_food_waste_auto(image, packaging_mask)
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
            st.image(qr_code_image
