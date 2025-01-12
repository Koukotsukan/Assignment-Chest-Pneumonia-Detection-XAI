from dash import Input, Output, State, html, dcc, ctx, no_update
import dash_bootstrap_components as dbc
from utils.grad_cam import generate_grad_cam, generate_grad_cam_plus, get_last_conv_layer, generate_grad_cam_explain
from utils.lime_utils import compute_shap_values, plot_shap
from utils.preprocess import preprocess_image, adjust_image
from utils.counterfactual import simulate_counterfactual, smooth_image
from models.model_loader import load_vgg16
import plotly.express as px
import numpy as np
from PIL import Image
import base64
import io

# 加载模型
model = load_vgg16()

def register_callbacks(app):
    # 文件上传状态和删除逻辑合并
    @app.callback(
        [Output("file-status", "children"),
         Output("upload-image", "contents")],
        [Input("upload-image", "filename"),
         Input("upload-image", "contents"),
         Input("remove-file-button", "n_clicks")],
        prevent_initial_call=True
    )
    # 在 handle_file_upload 中处理上传图片
    def handle_file_upload(filename, contents, remove_clicks):
        triggered_id = ctx.triggered_id

        # 文件删除
        if triggered_id == "remove-file-button":
            return None, None

        # 文件上传
        if filename and contents:
            # 对上传图片进行预处理和调整大小
            try:
                # contents = preprocess_and_resize_image_to_base64(contents)

                # 返回上传状态信息
                upload_status = dbc.Alert(
                    [
                        html.Span(f"Uploaded: {filename}"),
                        dbc.Progress(value=100, striped=True, animated=True, className="my-2"),
                        dbc.Button("Remove", id="remove-file-button", color="danger", size="sm", outline=True)
                    ],
                    color="info",
                    dismissable=False
                )
                return upload_status, contents
            except Exception as e:
                return dbc.Alert(f"Error processing the image: {str(e)}", color="danger"), None

        return no_update, no_update

    # 分析逻辑回调
    @app.callback(
        Output("output-container", "children"),
        Input("submit-button", "n_clicks"),
        [State("upload-image", "contents"),
         State("upload-image", "filename"),
         State("question-dropdown", "value")],
        prevent_initial_call=True
    )
    def update_output(n_clicks, image_content, filename, question_type):
        if n_clicks is None or not image_content or not question_type:
            return dbc.Alert("Please upload an image and select a question.", color="warning"), {"display": "none"}
        image_array = preprocess_image(image_content)

        if question_type == "shap":
            shap_values = compute_shap_values(model, image_array)
            fig = plot_shap(image_array, shap_values)
            result = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), width=6),
                dbc.Col(html.Img(src=image_content, style={"maxWidth": "100%"}), width=6)
            ])
        elif question_type == "grad_cam":
            last_layer = get_last_conv_layer(model)
            heatmap_1 = generate_grad_cam(image_array, model, last_layer)
            heatmap_2 = generate_grad_cam_explain(image_array, model)
            fig_1 = px.imshow(heatmap_1, color_continuous_scale="jet", title="Grad-CAM Heatmap")
            fig_2 = px.imshow(heatmap_2, color_continuous_scale="jet", title="Grad-CAM Heatmap Overlapped")
            fig_original = px.imshow(
                image_array.squeeze(),
                title="Original Image",
                color_continuous_scale="gray"
            )
            result = dbc.Row([
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            html.H5("Grad-CAM Explanation:"),
                            style={"textAlign": "center", "color": "#444", "marginBottom": "5px"}
                        ),
                        width=12
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            "If the Grad-CAM heatmap appears fully green, it may indicate the following:\n"
                            "- The model considers the input image unrelated to the target class.\n"
                            "- The input image lacks prominent features, or the features failed to activate in the selected convolutional layer.\n"
                            "- The model has not sufficiently learned this class, or the feature space is not well-defined.",
                            style={"textAlign": "center", "color": "#888", "fontSize": "14px"}
                        ),
                        width=12
                    )
                ),
                dbc.Col(dcc.Graph(figure=fig_original), width=6),
                dbc.Col(dcc.Graph(figure=fig_1), width=6),
                dbc.Col(dcc.Graph(figure=fig_2), width=6),
                # dbc.Col(html.Img(src=image_content, style={"maxWidth": "100%"}), width=6)


            ], style={"marginTop": "20px"}
            )
        elif question_type == "counterfactual":
            print(image_array)
            try:
                counterfactual_image = simulate_counterfactual(image_array, model, 1)
                counterfactual_image = smooth_image(counterfactual_image, sigma=0.5)
                # 确保 counterfactual_image 的形状正确
                if counterfactual_image.shape[0] == 1:  # 检查是否有批量维度
                    counterfactual_image = counterfactual_image[0]  # 移除第一个维度
            except Exception as e:
                return dbc.Alert(f"Error generating counterfactual: {str(e)}", color="danger"), {"display": "none"}
            print(f"Counterfactual image shape: {counterfactual_image.shape}")
            print(f"Original image shape: {image_array.squeeze().shape}")

            fig_counterfactual = px.imshow(
                counterfactual_image,
                title="Counterfactual Image",
                color_continuous_scale="viridis"
            )
            fig_original = px.imshow(
                image_array.squeeze(),
                title="Original Image",
                color_continuous_scale="gray"
            )
            result = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_original), width=6),
                dbc.Col(dcc.Graph(figure=fig_counterfactual), width=6),
            ])
        else:
            result = dbc.Alert("Unknown question type selected.", color="danger")
        return result

