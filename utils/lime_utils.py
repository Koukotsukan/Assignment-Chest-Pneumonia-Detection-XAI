from lime import lime_image
import numpy as np
import cv2
from skimage.segmentation import felzenszwalb, mark_boundaries
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


# 定义 Dice Coefficient 和 Dice Loss
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice Coefficient: 用于衡量分割的准确性。
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Dice Loss: Dice Coefficient 的损失版本。
    """
    return 1 - dice_coef(y_true, y_pred)
# 加载 U-Net 模型（全局加载，避免多次加载影响性能）
_unet_model = load_model("models/unet_lung_seg.hdf5",
                         custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef}
                         )


def segment_lungs_with_unet(image):
    """
    使用 U-Net 模型对输入图像进行肺部分割（支持 512x512 固定输入大小）。
    Args:
        image (numpy array): 输入图像 (height, width, channels)，范围 [0, 1]。
    Returns:
        lung_mask (numpy array): 二值化的肺部蒙版，形状为 (height, width)。
    """
    # 确保输入图像为 float32 类型并在 [0, 255] 范围内
    if image.dtype != np.uint8 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    elif image.dtype == np.float64:
        image = image.astype(np.uint8)

    # 确保输入图像为灰度图
    if len(image.shape) == 3 and image.shape[-1] == 3:  # 如果是 RGB 图像
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3 and image.shape[-1] == 4:  # 如果是 RGBA 图像
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # 调整图像大小到 U-Net 的输入尺寸 (512x512) 并扩展维度
    input_image = cv2.resize(image, (512, 512))  # 调整大小
    input_image = np.expand_dims(input_image, axis=-1)  # 添加单通道维度
    input_image = np.expand_dims(input_image, axis=0)  # 添加批次维度

    # 预测肺部蒙版
    lung_mask = _unet_model.predict(input_image)[0]

    # 调整蒙版大小回原始图像尺寸
    lung_mask = cv2.resize(lung_mask, (image.shape[1], image.shape[0]))
    lung_mask = (lung_mask > 0.5).astype(np.uint8)  # 二值化
    return lung_mask



def compute_lime(model, image_array):
    """
    使用 U-Net 生成的肺部蒙版优化 LIME 分析。
    Args:
        model: TensorFlow/Keras 模型。
        image_array: 输入图像，形状为 (1, height, width, channels)。
    Returns:
        explanation: LIME 分析结果。
    """
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    image = image_array[0]

    # 使用 U-Net 提取肺部蒙版
    lung_mask = segment_lungs_with_unet(image)

    # 自定义分割函数，仅在肺部区域内分割
    def custom_segmentation(image):
        segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=100)
        # 仅保留肺部区域的分割
        segments = np.where(lung_mask > 0, segments, -1)
        return segments

    # 定义预测函数
    def predict_fn(images):
        images = np.array(images) / 255.0  # 确保归一化到 [0, 1]
        predictions = model.predict(images)  # Softmax 输出
        if predictions.ndim != 2:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
        return predictions

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image, predict_fn, top_labels=2, hide_color=0,
        segmentation_fn=custom_segmentation, num_samples=1000
    )

    return explanation
