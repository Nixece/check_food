import tensorflow as tf
import numpy as np
import cv2

# โหลดโมเดล MobileNetV2 (เทรนบน ImageNet)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# ฟังก์ชันจำแนกรูปภาพ
def classify_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    return decoded  # คืนค่าเป็น (imagenet_id, label, confidence)
