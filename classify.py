import tensorflow as tf
import numpy as np

# โหลดโมเดล MobileNetV2 ที่ฝึกกับ TrashNet (ต้องมี model.h5 ในโฟลเดอร์เดียวกัน)
model = tf.keras.models.load_model("model.h5")

# คลาสที่ TrashNet ใช้
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# แปล label ภาษาอังกฤษ → ภาษาไทย + ประเภทขยะ
def translate_label(label):
    label_map = {
        'cardboard': ('กล่องกระดาษ', 'ขยะรีไซเคิล'),
        'glass': ('ขวดแก้ว', 'ขยะรีไซเคิล'),
        'metal': ('โลหะ / กระป๋อง', 'ขยะรีไซเคิล'),
        'paper': ('กระดาษ', 'ขยะรีไซเคิล'),
        'plastic': ('พลาสติก', 'ขยะรีไซเคิล'),
        'trash': ('ขยะทั่วไป', 'ขยะทั่วไป')
    }
    return label_map.get(label, ('ไม่ทราบ', 'ไม่ทราบ'))

# ฟังก์ชันทำนายประเภทขยะ
def classify_waste_image(image):
    # resize & normalize
    img = tf.image.resize(image, (224, 224)) / 255.0
    img = tf.expand_dims(img, axis=0)

    # ทำนาย
    predictions = model.predict(img)[0]
    class_index = np.argmax(predictions)
    confidence = float(predictions[class_index])
    label_en = class_labels[class_index]
    label_th, category_th = translate_label(label_en)

    return {
        "label_en": label_en,
        "label_th": label_th,
        "confidence": confidence,
        "category_th": category_th
    }
