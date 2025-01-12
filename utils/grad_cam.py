import numpy as np
from keras.layers import Conv2D
import streamlit as st

def generate_grad_cam_explain(img_array, model, layer_name, target_class=None):
    """
    Grad-CAM explain 实现，用于二分类 Softmax 模型。
    Args:
        img_array: 输入图像，形状为 (1, height, width, channels)。
        model: 目标模型。
        layer_name: 用于 Grad-CAM 的卷积层名称。
        target_class: 目标类别（0 或 1）。如果为 None，则自动选择 Softmax 概率最大的类别。
    Returns:
        numpy array: Grad-CAM 热图叠加结果。
    """
    from tf_explain.core.grad_cam import GradCAM

    explainer = GradCAM()

    # 预测类别概率
    predictions = model.predict(img_array)
    if target_class is None:
        target_class = np.argmax(predictions[0])  # 动态选择类别（0 或 1）

    # st.info(f"Predictions: {predictions[0]} | Target Class: {target_class}")
    # st.markdown(
    #     "**Chest X-Ray Pneumonia Prediction:** " + str(
    #         "NORMAL" if predictions[0, 0] < 0.5 else "PNEUMONIA"))
    # st.markdown("**Prediction Confidence:** " + str(round(predictions[0, 0], 4)))
    # 数据准备
    data = (img_array, target_class)

    # 生成 Grad-CAM
    explanation = explainer.explain(data, model, target_class)
    heatmap = explanation / 255.0  # 归一化到 [0, 1]

    return heatmap
def get_last_conv_layer(model):
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            last_conv_layer = layer
    return last_conv_layer

def get_second_last_conv_layer(model):
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    if len(conv_layers) < 2:
        return None  # 如果模型少于两个卷积层，则返回 None
    return conv_layers[-2]  # 返回倒数第二个卷积层