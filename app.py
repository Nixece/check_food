import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os

# ฟังก์ชันรีเซ็ตคะแนน (ลบไฟล์ scores.csv)
def reset_scores():
    if os.path.exists('scores.csv'):
        os.remove('scores.csv')
        st.write("คะแนนสะสมทั้งหมดถูกรีเซ็ตแล้ว!")

# ฟังก์ชันแปลงภาพเป็นรูปแบบสี HSV เพื่อใช้ในการตรวจจับสีที่คล้ายอาหาร
def is_food_basic(image):
    # แปลงภาพเป็นสี HSV
    img_array = np.array(image)
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # กำหนดขอบเขตของสีที่พบบ่อยในอาหาร (เช่น สีน้ำตาล, สีขาว)
    lower_brown = np.array([10, 50, 20])  # ขอบเขตล่างของสีน้ำตาล
    upper_brown = np.array([20, 255, 200])  # ขอบเขตบนของสีน้ำตาล

    lower_white = np.array([0, 0, 200])  # ขอบเขตล่างของสีขาว
    upper_white = np.array([180, 30, 255])  # ขอบเขตบนของสีขาว
    
    # สร้าง mask สำหรับสีน้ำตาลและสีขาว
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # รวม mask ทั้งสองเข้าด้วยกัน
    combined_mask = cv2.bitwise_or(mask_brown, mask_white)
    
    # คำนวณจำนวนพิกเซลที่ตรงกับสีขาวหรือน้ำตาล
    food_pixels = cv2.countNonZero(combined_mask)
    
    # ตรวจสอบสัดส่วนของพิกเซลอาหารกับพิกเซลทั้งหมดในภาพ
    total_pixels = img_array.shape[0] * img_array.shape[1]
    food_ratio = food_pixels / total_pixels
    
    # ถ้าสัดส่วนพิกเซลอาหารมากกว่า 10% ถือว่าเป็นภาพอาหาร
    if food_ratio > 0.1:
        return True
    return False

# ฟังก์ชันการคำนวณคะแนนจากปริมาณเศษอาหารที่เหลือในจาน
def calculate_waste(image):
    gray_image = image.convert("L")  # แปลงภาพเป็นขาวดำ
    img_array = np.array(gray_image)

    # ใช้การทำ Threshold เพื่อแยกเศษอาหารออกจากพื้นหลัง
    _, threshold_image = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
    
    # ค้นหา Contours
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # คำนวณจำนวนพิกเซลทั้งหมดในภาพ
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    # คำนวณพื้นที่ที่เป็นเศษอาหาร
    waste_pixels = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1000)

    # คำนวณสัดส่วนพื้นที่เศษอาหารที่เหลือ
    waste_ratio = waste_pixels / total_pixels
    
    # ให้คะแนนตามสัดส่วนเศษอาหาร
    if waste_ratio > 0.5:
        return 0
    elif waste_ratio < 0.05:
        return 10
    return int((1 - waste_ratio) * 10)

# ฟังก์ชันสำหรับการบันทึกคะแนน
def save_score(student_id, score):
    if not os.path.exists('scores.csv'):
        df = pd.DataFrame(columns=['Student ID', 'Score'])
    else:
        df = pd.read_csv('scores.csv')
    
    new_entry = pd.DataFrame([{'Student ID': student_id, 'Score': score}])
    df = pd.concat([df, new_entry], ignore_index=True)
    st.write(f"เพิ่มนักศึกษาหมายเลข {student_id} พร้อมคะแนน: {score}")
    
    df.to_csv('scores.csv', index=False)

# ฟังก์ชันแสดง 10 คนล่าสุด
def show_latest_entries():
    if os.path.exists('scores.csv'):
        df = pd.read_csv('scores.csv')
        latest_10 = df.tail(10)
        st.write("📋 **10 คนล่าสุดที่ส่งข้อมูล** 📋")
        st.table(latest_10)
    else:
        st.write("ยังไม่มีข้อมูลคะแนน")

# ส่วนของการใส่เลขประจำตัวนักศึกษาและอัปโหลดรูป
st.title('Food Waste LE02')
st.write("กรอกเลขประจำตัวนักศึกษาและอัปโหลดภาพจานอาหารเพื่อรับคะแนน")

# ปุ่มรีเซ็ตคะแนน
if st.button('รีเซ็ตคะแนน'):
    reset_scores()

# ใส่เลขประจำตัวนักศึกษา
student_id = st.text_input("ใส่เลขประจำตัวนักศึกษา")

# ส่วนการอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกภาพที่ต้องการอัปโหลด", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and student_id:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="ภาพที่คุณอัปโหลด", use_column_width=True)
    
    # ตรวจสอบว่าเป็นอาหารหรือไม่
    if not is_food_basic(image):
        st.write("กรุณาอัปโหลดภาพที่เป็นอาหารเท่านั้น")
    else:
        # คำนวณคะแนน
        score = calculate_waste(image)
        st.write(f"คะแนนที่ได้: {score} / 10")
        
        # บันทึกคะแนน
        save_score(student_id, score)

# แสดง 10 คนล่าสุดที่ส่งข้อมูล
show_latest_entries()
