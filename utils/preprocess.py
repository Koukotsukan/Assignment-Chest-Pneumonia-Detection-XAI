import cv2
import base64
import io
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array

def preprocess_image(base64_image, target_size=(150, 150)):
    # 解码 Base64 数据
    base64_str = base64_image.split(",")[1]
    image_data = base64.b64decode(base64_str)

    # 加载图像并调整大小
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)

    # 归一化并添加批量维度
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

def adjust_image(image_array, brightness, contrast):
    return cv2.convertScaleAbs(image_array, alpha=contrast, beta=brightness)
