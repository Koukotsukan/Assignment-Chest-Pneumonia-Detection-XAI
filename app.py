import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
from keras import Model
from skimage.segmentation import mark_boundaries
from utils.grad_cam import generate_grad_cam_explain
from utils.lime_utils import compute_lime
from utils.counterfactual import simulate_counterfactual, smooth_image
from models.model_loader import load_vgg16, load_cnn, load_mobilenet
import tensorflow as tf

# 初始化 session_state
if "cnn_model" not in st.session_state:
    st.session_state.cnn_model = load_cnn()
if "vgg_model" not in st.session_state:
    st.session_state.vgg_model = load_vgg16()
if "mobilenet_model" not in st.session_state:
    st.session_state.mobilenet_model = load_mobilenet()
if "resized_size_cnn" not in st.session_state:
    st.session_state.resized_size_cnn = (256, 256)
if "resized_size_vgg" not in st.session_state:
    st.session_state.resized_size_vgg = (150, 150)
if "resized_size_mobilenet" not in st.session_state:
    st.session_state.resized_size_mobilenet = (224, 224)
if "layer_name_vgg" not in st.session_state:
    st.session_state.layer_name_vgg = "block5_conv3"
if "layer_name_mobilenet" not in st.session_state:
    st.session_state.layer_name_mobilenet = "block_15_project"


st.set_page_config(
    page_title="Chest X-rays Analysis",  # 页面标题
    page_icon="🩺",  # 页面图标
    layout="centered",  # 页面布局
    initial_sidebar_state="expanded",  # 侧边栏初始状态
)

st.title("Question-Driven Pneumonia XAI Dashboard")
st.image(
    "assets/banner.png",
    use_container_width=True,
)

# 文件上传
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

# 分析类型选择
analysis_type = st.selectbox(
    "Select a question to explore:",
    options=["What (VGG16-FineTune)", "Why (LIME+Mobile-Net-V2-FinTune)", "Where (GradCAM+Mobile-Net-V2-FineTune)", "Why Not (Counterfactual+VGG-FineTune)"]
)

# 点击 Analyze 按钮
if st.button("Analyze"):
    if uploaded_file is not None:
        # 加载图片
        image = Image.open(uploaded_file).convert("RGB")

        if analysis_type == "Why (LIME+Mobile-Net-V2-FinTune)":
            # 使用 CNN
            image = image.resize(st.session_state.resized_size_mobilenet)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing with Lime..."):
                try:
                    # 调用优化的 LIME 分析函数
                    explanation = compute_lime(st.session_state.mobilenet_model, image_array)

                    # 获取 LIME 的叠加图像和掩膜
                    lime_image, mask = explanation.get_image_and_mask(
                        label=explanation.top_labels[0], positive_only=True, num_features=25, hide_rest=False
                    )
                    lime_overlay = mark_boundaries(lime_image, mask)

                    # 使用 Plotly 显示结果
                    original_fig = px.imshow(image_array[0], title="Original Image")
                    original_fig.update_layout(coloraxis_showscale=False)
                    original_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

                    lime_fig = px.imshow(lime_overlay, title="LIME Explanation (With Lung Focus)")
                    lime_fig.update_layout(coloraxis_showscale=False)
                    lime_fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

                    # 显示图像
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.plotly_chart(original_fig, use_container_width=True)
                    with col2:
                        st.subheader("LIME Explanation")
                        st.plotly_chart(lime_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during LIME analysis: {str(e)}")

        elif analysis_type == "Where (GradCAM+Mobile-Net-V2-FineTune)":
                image = image.resize((224,224))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                with st.spinner("Analyzing with Multiple Grad-CAMs..."):
                    predictions = st.session_state.vgg_model.predict(image_array)
                    prediction_confidences = [float(value) for value in predictions[0]]
                    target_class = np.argmax(predictions[0])
                    target_class_confidence = prediction_confidences[target_class]
                    st.markdown(
                        "**Chest X-Ray Pneumonia Prediction:** " + str("NORMAL" if target_class < 0.5 else "PNEUMONIA"))
                    st.markdown("**Prediction Confidence:** " + str(round(target_class_confidence, 4)))
                    heatmap_explain = generate_grad_cam_explain(
                        image_array, st.session_state.mobilenet_model, st.session_state.layer_name_mobilenet
                    )
                    # 可视化
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        original_fig = px.imshow(image_array[0], color_continuous_scale="viridis")
                        original_fig.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(original_fig, use_container_width=True)
                    with col2:
                        st.subheader("Overlapped")
                        explain_fig = px.imshow(heatmap_explain)
                        st.plotly_chart(explain_fig, use_container_width=True)
                    # except Exception as e:
                    #     st.error(f"Error in Grad-CAM analysis: {str(e)}")

        elif analysis_type == "Why Not (Counterfactual+VGG-FineTune)":
            # 使用 VGG16
            image = image.resize(st.session_state.resized_size_vgg)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing with Counterfactuals..."):
                try:
                    original_predictions = st.session_state.vgg_model(image_array)
                    target_proba = 1 - original_predictions[0, 0]  # Generate counterfactual for opposite class
                    st.markdown(f"**Original Prediction:** {'NORMAL' if original_predictions[0, 0] < 0.5 else 'PNEUMONIA'}")
                    counterfactual_image = simulate_counterfactual(image_array, st.session_state.vgg_model, target_proba)
                    counterfactual_image = smooth_image(counterfactual_image, sigma=0.5)
                    try:
                        pred_cf = st.session_state.vgg_model(counterfactual_image)
                        st.markdown("**Counterfactual Prediction:** " + str("NORMAL" if pred_cf[0,0] < 0.5 else "PNEUMONIA"))
                    except:
                        st.text("")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        original_fig = px.imshow(image_array[0], color_continuous_scale="viridis")
                        st.plotly_chart(original_fig, use_container_width=True)
                    with col2:
                        st.subheader("Counterfactual Image")
                        counter_fig = px.imshow(counterfactual_image[0], color_continuous_scale="viridis")
                        st.plotly_chart(counter_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating counterfactual: {str(e)}")
        elif analysis_type == "What (VGG16-FineTune)":
            image = image.resize(st.session_state.resized_size_vgg)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            with st.spinner("Analyzing with VGG16-FineTune..."):
                # try:
                predictions = st.session_state.vgg_model.predict(image_array)
                prediction_confidences = [float(value) for value in predictions[0]]
                target_class = np.argmax(predictions[0])
                target_class_confidence = prediction_confidences[target_class]
                st.markdown(
                    "**Chest X-Ray Pneumonia Prediction:** " + str("NORMAL" if target_class < 0.5 else "PNEUMONIA"))
                st.markdown("**Prediction Confidence:** " + str(round(target_class_confidence, 4)))
                st.subheader("Uploaded Image")
                original_fig = px.imshow(image_array[0], color_continuous_scale="viridis")
                st.plotly_chart(original_fig, use_container_width=True)
                # except Exception as e:
                #     st.error(f"Error generating prediction: {str(e)}")

    else:
        st.warning("Please upload an image.")
def add_footer():
    footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f5f5f5;
            padding: 10px 0;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
        }
    </style>
    <div class="footer">
        © 2025 Niu Zhaohang S2001904/2, Universiti Malaya | Data from <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank">Chest X-ray Pneumonia Dataset by Kaggle @paultimothymooney</a>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

add_footer()