import streamlit as st

# ฟังก์ชันเพื่อแยกประเภทขยะ
def classify_waste(item):
    # Dictionary สำหรับประเภทขยะ
    waste_types = {
        "ขยะทั่วไป": [
            "ถุงพลาสติก", "ขวดพลาสติกที่ใช้ครั้งเดียว", "ซองขนม", "ฟอยล์", "ถุงขนม", 
            "หลอดพลาสติก", "เศษอาหารที่ไม่สามารถย่อยสลาย", "โฟม", "บรรจุภัณฑ์พลาสติก", "เทปกาว"
        ],
        "ขยะรีไซเคิล": [
            "ขวดพลาสติก", "กระป๋อง", "กระดาษ", "กระดาษกล่อง", "ขวดแก้ว", 
            "กระดาษหนังสือพิมพ์", "กล่องกระดาษ", "กระดาษทิชชูที่ไม่ใช้แล้ว", "ถุงรีไซเคิล"
        ],
        "ขยะเปียก": [
            "เศษอาหาร", "เปลือกผลไม้", "ใบไม้", "อาหารเหลือ", "กาแฟที่เหลือ", 
            "น้ำผลไม้ที่เหลือ", "ทิชชูเปียก", "เนื้อสัตว์", "กระดูก"
        ]
    }

    # ตรวจสอบว่า item อยู่ในประเภทใด
    for waste_type, items in waste_types.items():
        if item in items:
            return waste_type
    return "ไม่ทราบประเภท"

# ส่วนของอินพุตใน Streamlit
st.title("♻️ โปรแกรมแยกขยะอัจฉริยะ")
st.write("กรุณากรอกสิ่งของที่ต้องการทิ้ง (หนึ่งรายการต่อบรรทัด)")

# รับข้อมูลจากผู้ใช้
input_items = st.text_area("🗑️ กรอกขยะที่ต้องการทิ้ง", 
                          help="พิมพ์สิ่งของ เช่น ขวดพลาสติก, เศษอาหาร, กระดาษ ฯลฯ")

# หากมีการกรอกข้อมูล
if input_items:
    items = [item.strip().lower() for item in input_items.split("\n") if item.strip()]
    
    categorized_waste = {"ขยะทั่วไป": [], "ขยะรีไซเคิล": [], "ขยะเปียก": [], "ไม่ทราบประเภท": []}

    for item in items:
        waste_type = classify_waste(item)
        categorized_waste[waste_type].append(item)

    # แสดงผลลัพธ์แบบ Markdown
    st.subheader("📋 ผลลัพธ์การแยกขยะ")
    
    for waste_type, items in categorized_waste.items():
        if items:
            st.markdown(f"**{waste_type}**: {', '.join(items)}")
        else:
            st.markdown(f"**{waste_type}**: ไม่มีขยะประเภทนี้")

