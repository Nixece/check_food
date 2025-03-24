import streamlit as st

# ฟังก์ชันแยกประเภทขยะ
def classify_waste(item):
    waste_types = {
        "ขยะทั่วไป": ["ถุงพลาสติก", "โฟม", "เทปกาว", "หลอดพลาสติก"],
        "ขยะรีไซเคิล": ["ขวดพลาสติก", "กระป๋อง", "กระดาษ", "ขวดแก้ว"],
        "ขยะเปียก": ["เศษอาหาร", "เปลือกผลไม้", "ใบไม้", "เนื้อสัตว์"]
    }
    for waste_type, items in waste_types.items():
        if item in items:
            return waste_type
    return "ไม่ทราบประเภท"

# UI Streamlit
st.title("♻️ โปรแกรมแยกขยะอัจฉริยะ")
st.write("กรุณากรอกสิ่งของที่ต้องการทิ้ง (หนึ่งรายการต่อบรรทัด)")

# รับข้อมูล
input_items = st.text_area("🗑️ กรอกขยะที่ต้องการทิ้ง", help="พิมพ์ชื่อขยะ เช่น ขวดพลาสติก, เศษอาหาร")

if input_items:
    with st.spinner("⏳ กำลังประมวลผล..."):
        items = [item.strip().lower() for item in input_items.split("\n") if item.strip()]
        categorized_waste = [{"ขยะ": item, "ประเภท": classify_waste(item)} for item in items]
    
    # แสดงผลแบบตารางที่โหลดเร็วขึ้น
    st.subheader("📋 ผลลัพธ์การแยกขยะ")
    st.table(categorized_waste)
