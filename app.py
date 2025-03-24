import streamlit as st

# ฟังก์ชันเพื่อแยกประเภทขยะ
def classify_waste(item):
    # สร้าง dictionary สำหรับประเภทขยะ
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

    # สร้างผลลัพธ์สำหรับขยะที่พบ
    result = {"ขยะทั่วไป": [], "ขยะรีไซเคิล": [], "ขยะเปียก": []}
    
    # ถ้า item อยู่ในประเภทไหนก็เพิ่มไปใน dictionary
    for waste_type, items in waste_types.items():
        if item in items:
            result[waste_type].append(item)
    
    return result

# แสดงส่วนของอินพุตใน Streamlit
st.title("โปรแกรมแยกขยะสำหรับนักศึกษา")
st.write("กรุณากรอกสิ่งที่คุณต้องการทิ้ง")

# ให้ผู้ใช้กรอกสิ่งที่ต้องการทิ้ง
input_items = st.text_area("กรอกสิ่งที่ต้องการทิ้ง (ใส่สิ่งของที่ต้องการแยกแต่ละชิ้นในบรรทัดใหม่)", 
                          help="กรอกขยะที่ต้องการทิ้ง เช่น ขวดพลาสติก, เศษอาหาร, กระดาษ, หรือขยะอื่นๆ")

# แยกสิ่งของจากการกรอกข้อมูล
if input_items:
    items = input_items.split("\n")  # แยกตามบรรทัด
    
    # สร้างผลลัพธ์ของการแยกประเภทขยะ
    categorized_waste = {"ขยะทั่วไป": [], "ขยะรีไซเคิล": [], "ขยะเปียก": []}
    
    for item in items:
        item = item.strip()  # กำจัดช่องว่าง
        result = classify_waste(item)
        
        for waste_type, categorized_items in result.items():
            categorized_waste[waste_type].extend(categorized_items)
    
    # แสดงผลลัพธ์ที่แยกประเภทขยะ
    st.write("ผลลัพธ์การแยกขยะ:")

    for waste_type, items in categorized_waste.items():
        if items:
            st.write(f"- {waste_type}: {', '.join(items)}")
        else:
            st.write(f"- {waste_type}: ไม่มีขยะประเภทนี้ในรายการของคุณ")
