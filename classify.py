import tensorflow as tf
import numpy as np
import cv2

# โหลด MobileNetV2 ที่ฝึกกับ ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# ฟังก์ชันทำนายภาพ → label
def classify_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    return decoded  # (imagenet_id, label, confidence)

# Mapping label → ประเภทขยะ 5 ประเภท
def map_label_to_waste_category(label):
    label = label.lower()

    recycle = ['bottle', 'can', 'carton', 'glass', 'paper', 'plastic', 'tin', 'box']
    organic = ['banana', 'apple', 'orange', 'food', 'meat', 'vegetable', 'fruit', 'corn']
    general = ['trash', 'sponge', 'diaper', 'tissue', 'wrapper']
    hazardous = ['battery', 'lighter', 'syringe', 'medication', 'chemical']
    electronic = ['cellular_telephone', 'remote', 'tv', 'computer', 'keyboard', 'mouse']

    if any(word in label for word in recycle):
        return 'ขยะรีไซเคิล'
    elif any(word in label for word in organic):
        return 'ขยะเปียก'
    elif any(word in label for word in hazardous):
        return 'ขยะอันตราย'
    elif any(word in label for word in electronic):
        return 'ขยะอิเล็กทรอนิกส์'
    else:
        return 'ขยะทั่วไป'
