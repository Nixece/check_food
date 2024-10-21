import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
from PIL import ImageOps

# ฟังก์ชันรีเซ็ตคะแนน (ลบไฟล์ scores.csv)
def reset_scores():
    try:
        if os.path.exists('scores.csv'):
            os.remove('scores.csv')
            st.write("คะแนนสะสมทั้งหมดถูกรีเซ็ตแล้ว!")
        else:
            st.write("ยังไม่มีคะแนนที่ถูกบันทึก")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรีเซ็ตคะแนน: {e}")

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

# ฟังก์ชันการคำนวณคะแนนจากปริมาณเศษอาหารที่เหลือในจาน
def calculate_waste(image):
    try:
        gray_image = image.convert("L")  # แปลงภาพเป็นขาวดำ
        img_array = np.array(gray_image)

        # ใช้การทำ Threshold เพื่อแยกเศษอาหารออกจากพื้นหลัง
        _, threshold_image = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
        
        # ค้นหา Contours
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # คำนวณจำนวนพิกเซลทั้งหมดในภาพ
        total_pixels = img_array.size

        # คำนวณจำนวนพิกเซลของเศษอาหาร (ใช้การหาพื้นที่ Contours)
        waste_pixels = sum(cv2.contourArea(contour) for contour in contours)
        
        # คำนวณสัดส่วนพื้นที่เศษอาหารที่เหลือ
        waste_ratio = waste_pixels / total_pixels

        # เกณฑ์การให้คะแนนใหม่:
        if waste_ratio > 0.5:
            return np.random.randint(0, 3)  # ถ้าเศษอาหารเกิน 50% ได้ 0-2 คะแนน
        elif waste_ratio < 0.1:
            return 10  # ถ้าเศษอาหารน้อยกว่า 10% ได้ 10 คะแนน
        else:
            return np.random.randint(5, 8)  # ถ้าเศษอาหารระหว่าง 10%-50% ได้ 5-7 คะแนน
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return 0

# ฟังก์ชันสำหรับการบันทึกคะแนน
def save_score(name, score):
    try:
        if not os.path.exists('scores.csv'):
            df = pd.DataFrame(columns=['Name', 'Score'])
        else:
            df = pd.read_csv('scores.csv')
        
        new_entry = pd.DataFrame([{'Name': name, 'Score': score}])
        df = pd.concat([df, new_entry], ignore_index=True)
        st.write(f"เพิ่ม {name} พร้อมคะแนน: {score}")
        
        df.to_csv('scores.csv', index=False)
    except PermissionError:
        st.error("ไม่สามารถบันทึกคะแนนได้ ไฟล์กำลังถูกใช้งาน ลองใหม่อีกครั้ง")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการบันทึกคะแนน: {e}")

# ฟังก์ชันแสดง 10 คนล่าสุด
def show_latest_entries():
    if os.path.exists('scores.csv'):
        df = pd.read_csv('scores.csv')
        latest_10 = df.tail(10)
        st.write("📋 **10 คนล่าสุดที่ส่งข้อมูล** 📋")
        st.table(latest_10)
    else:
        st.write("ยังไม่มีข้อมูลคะแนน")

# ส่วนของการใส่ชื่อและอัปโหลดรูป
st.title('📉 Food Waste Score System')
st.write("""
    🚨 **คำแนะนำ**: กรุณากรอกชื่อและอัปโหลดภาพอาหารที่ชัดเจน เพื่อรับคะแนนประเมินเศษอาหาร 
    \nคะแนนจะประเมินตามปริมาณเศษอาหารที่เหลือในจาน
""")

# ปุ่มรีเซ็ตคะแนนพร้อมการตรวจสอบรหัสผ่าน
with st.expander("รีเซ็ตคะแนน (สำหรับผู้ดูแลระบบเท่านั้น)"):
    reset_password = st.text_input("กรอกรหัสผ่านเพื่อรีเซ็ตคะแนน", type="password")
    if st.button('รีเซ็ตคะแนน'):
        if reset_password == "LE02":  # ตรวจสอบรหัสผ่าน
            reset_scores()
        else:
            st.error("รหัสผ่านไม่ถูกต้อง")

# ใส่ชื่อผู้ใช้
name = st.text_input("ใส่ชื่อผู้ใช้งาน")

# ส่วนการอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกภาพที่ต้องการอัปโหลด", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and name:
    image = Image.open(uploaded_file)
    
    # ปรับขนาดภาพ
    image = resize_image(image)

    st.image(image, caption="ภาพที่คุณอัปโหลด", use_column_width=True)
    
    # ตรวจสอบว่าเป็นอาหารหรือไม่
    if not is_food_basic(image):
        st.write("กรุณาอัปโหลดภาพที่เป็นอาหารเท่านั้น")
    else:
        # คำนวณคะแนน
        score = calculate_waste(image)
        st.write(f"คะแนนที่ได้: {score} / 10")
        
        # บันทึกคะแนน
        save_score(name, score)

# แสดง 10 คนล่าสุดที่ส่งข้อมูล
show_latest_entries()
