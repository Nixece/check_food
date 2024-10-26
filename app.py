import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import ImageOps
import qrcode
import io

# ฟังก์ชันย่อขนาดภาพ
def resize_image(image, max_size=(300, 300)):
    return ImageOps.contain(image, max_size)

# ใช้ `st.cache_data` เพื่อเร่งความเร็วในการแปลงภาพเป็น numpy array
@st.cache_data
def load_image(image_file):
    return np.array(Image.open(image_file))

# ฟังก์ชันหาค่าความแตกต่างของสี
def color_difference(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))

# ฟังก์ชันหาสีเฉลี่ยจากขอบของบรรจุภัณฑ์
def get_edge_color_average(image, mask):
    edges = cv2.Canny(mask, 100, 200)
    edge_pixels = image[edges > 0]
    if len(edge_pixels) > 0:
        return np.mean(edge_pixels, axis=0)
    else:
        return np.array([0, 0, 0])  # เผื่อไว้ในกรณีที่ไม่มีพิกเซลที่ขอบ

# ฟังก์ชันหาความแตกต่างระหว่างพื้นหลังและบรรจุภัณฑ์
def detect_package(background, package_image):
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    package_gray = cv2.cvtColor(package_image, cv2.COLOR_RGB2GRAY)
    
    # หาความแตกต่างระหว่างภาพพื้นหลังและภาพบรรจุภัณฑ์
    diff = cv2.absdiff(background_gray, package_gray)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # กรอง Contours เล็ก ๆ ออก
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # กรองพื้นที่ขนาดเล็กออก
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
    
    # ลบพื้นหลังออกจากบรรจุภัณฑ์
    package_detected = cv2.bitwise_and(package_image, package_image, mask=mask)
    
    # คำนวณพื้นที่บรรจุภัณฑ์จากภาพที่เหลืออยู่หลังลบพื้นหลัง
    total_pixels = cv2.countNonZero(mask)
    
    return package_detected, total_pixels, mask

# ฟังก์ชันสำหรับตรวจจับเศษอาหารโดยใช้สีขอบบรรจุภัณฑ์เป็นตัวเทียบ
def check_food_waste_auto(image, mask, edge_color):
    try:
        # แปลงภาพเป็น RGB เพื่อเปรียบเทียบสี
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ตรวจสอบพิกเซลใน Mask ที่แตกต่างจากสีขอบ
        threshold = 30  # กำหนดเกณฑ์ความแตกต่างของสี
        food_waste_mask = np.zeros_like(mask)
        
        for y in range(image_rgb.shape[0]):
            for x in range(image_rgb.shape[1]):
                if mask[y, x] > 0:  # ตรวจเฉพาะพิกเซลในพื้นที่บรรจุภัณฑ์
                    pixel_color = image_rgb[y, x]
                    if color_difference(pixel_color, edge_color) > threshold:
                        food_waste_mask[y, x] = 255
        
        # คำนวณจำนวนพิกเซลของเศษอาหาร
        waste_pixels = cv2.countNonZero(food_waste_mask)
        
        # คำนวณเปอร์เซ็นต์ของเศษอาหาร
        waste_ratio = waste_pixels / cv2.countNonZero(mask)
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
    package_image = load_image(package_file)
    package_image = resize_image(Image.fromarray(package_image))
    package_array = np.array(package_image)
    
    # ตรวจจับบรรจุภัณฑ์และลบพื้นหลังหากมีภาพพื้นหลัง
    if background_file is not None:
        background_image = load_image(background_file)
        background_image = resize_image(Image.fromarray(background_image))
        background_array = np.array(background_image)
        
        # ตรวจจับบรรจุภัณฑ์และคำนวณพื้นที่บรรจุภัณฑ์
        package_detected, total_pixels, mask = detect_package(background_array, package_array)
        st.image(package_detected, caption="บรรจุภัณฑ์ที่ตรวจจับได้", use_column_width=True)

        # หาสีเฉลี่ยของขอบบรรจุภัณฑ์
        edge_color = get_edge_color_average(package_array, mask)
    else:
        package_detected = package_array
        total_pixels = cv2.countNonZero(cv2.cvtColor(package_detected, cv2.COLOR_RGB2GRAY))
        mask = cv2.inRange(cv2.cvtColor(package_detected, cv2.COLOR_RGB2GRAY), 1, 255)
        edge_color = get_edge_color_average(package_array, mask)
        st.image(package_detected, caption="ภาพที่มีบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติ
    result, passed = check_food_waste_auto(package_detected, mask, edge_color)
    st.write(result)

    # หากผ่านการตรวจสอบว่าไม่เหลืออาหาร
    if passed:
        st.success("บรรจุภัณฑ์นี้ไม่เหลือเศษอาหาร รับ 10 คะแนน!")
        
        # สร้าง QR Code ที่มีข้อมูลเฉพาะ
        qr_code_image = generate_qr_code("รหัสบรรจุภัณฑ์นี้สำหรับสะสม 10 คะแนน")
        
        # แสดง QR Code
        st.image(qr_code_image, caption="QR Code สำหรับสะสมแต้ม", use_column_width=False)
