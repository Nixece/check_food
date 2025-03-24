import tensorflow as tf
import numpy as np
import cv2

# โหลด MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# แปลง label → ประเภทขยะ 3 ประเภท
def map_label_to_3waste_categories(label):
    label = label.lower()
    recyclable = ['bottle', 'can', 'carton', 'box', 'paper', 'plastic', 'glass', 'newspaper', 'tin']
    organic = ['banana', 'apple', 'orange', 'vegetable', 'food', 'meat', 'fruit', 'carrot', 'corn']
    
    if any(word in label for word in recyclable):
        return 'ขยะรีไซเคิล'
    elif any(word in label for word in organic):
        return 'ขยะเปียก'
    else:
        return 'ขยะทั่วไป'

# ฟังก์ชันประมวลผลภาพและทำนาย
def classify_waste_image(image):
    img = cv2.resize(image, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    imagenet_id, label, score = decoded
    category = map_label_to_3waste_categories(label)

    return label, score, category
