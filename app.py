
import streamlit as st
import random

# สร้าง UI ของ Streamlit
st.title('Food Waste Analyzer')
st.write("อัปโหลดภาพจานอาหารของคุณเพื่อให้ AI ประเมินเศษข้าวที่เหลือ")

# ฟังก์ชันการให้คะแนนเศษข้าว (ให้คะแนนแบบสุ่มเพื่อจำลองการทำงานของ AI)
def predict_waste():
    return random.randint(0, 10)  # ให้คะแนน 0 ถึง 10 แบบสุ่ม

# ส่วนการอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกภาพที่ต้องการอัปโหลด", type="jpg")

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    st.image(uploaded_file, caption="ภาพที่คุณอัปโหลด", use_column_width=True)
    
    # แสดงคะแนนจาก AI
    score = predict_waste()
    st.write(f"คะแนนที่ได้: {score} / 10")
