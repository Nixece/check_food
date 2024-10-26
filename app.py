# ฟังก์ชันตรวจจับขนาดบรรจุภัณฑ์โดยอัตโนมัติ
def detect_package_size(image):
    # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
    image_array = np.array(image)
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # ใช้ Gaussian Blur เพื่อลด noise
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # ตรวจจับขอบของภาพโดยใช้ Canny Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # ค้นหา Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # หาพื้นที่ของ Contour ที่ใหญ่ที่สุด (ซึ่งน่าจะเป็นบรรจุภัณฑ์)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area

    # คืนค่าพื้นที่ของบรรจุภัณฑ์ (หน่วยเป็นพิกเซล)
    return max_area

# ฟังก์ชันสำหรับตรวจจับเศษอาหารและขนาดบรรจุภัณฑ์
def check_food_waste_auto_with_package(image):
    try:
        # ตรวจจับขนาดบรรจุภัณฑ์โดยอัตโนมัติ
        package_area = detect_package_size(image)

        # แปลงภาพเป็น numpy array และแปลงเป็นภาพขาวดำ
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ใช้ Gaussian Blur เพื่อลด noise
        blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

        # ใช้ Adaptive Threshold เพื่อตรวจจับเศษอาหาร
        threshold_image = cv2.adaptiveThreshold(blurred_image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # ค้นหา Contours ของเศษอาหาร
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waste_pixels = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # กรองสิ่งที่ไม่ใช่เศษอาหาร
                continue
            waste_pixels += area

        # คำนวณสัดส่วนของเศษอาหารเมื่อเทียบกับขนาดบรรจุภัณฑ์ที่ตรวจจับได้
        if package_area > 0:
            waste_ratio = waste_pixels / package_area
            waste_percentage = waste_ratio * 100
        else:
            waste_percentage = 100  # หากไม่พบขนาดบรรจุภัณฑ์ ให้ตั้งค่าเป็น 100%

        # แสดงผลลัพธ์
        if waste_ratio < 0.05:
            return f"บรรจุภัณฑ์ไม่เหลืออาหารเลย ({waste_percentage:.2f}%)", True
        else:
            return f"ยังเหลืออาหารอยู่ ({waste_percentage:.2f}%)", False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเศษอาหาร: {e}")
        return "เกิดข้อผิดพลาด", False

# ใช้งานโค้ดกับภาพที่อัปโหลด
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = resize_image(image)
    st.image(image, caption="ภาพบรรจุภัณฑ์", use_column_width=True)

    # ประเมินว่ามีเศษอาหารเหลืออยู่หรือไม่โดยอัตโนมัติพร้อมตรวจจับขนาดบรรจุภัณฑ์
    result, passed = check_food_waste_auto_with_package(image)
    st.write(result)
