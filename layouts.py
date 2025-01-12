from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    return html.Div(
        style={"margin": "0 auto", "maxWidth": "1200px", "padding": "20px"},
        children=[
            # 标题
            html.Header(
                html.Div(
                    "Comprehensive XAI Dashboard",
                    style={
                        "fontFamily": "'Montserrat', sans-serif",
                        "fontSize": "28px",
                        "fontWeight": "bold",
                        "textAlign": "left",
                        "marginBottom": "20px",
                        "color": "#333",
                    },
                )
            ),

            # 文件上传区域
            html.Div(
                [
                    dcc.Upload(
                        id="upload-image",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select a File")],
                            style={
                                "lineHeight": "40px",
                                "textAlign": "center",
                                "cursor": "pointer",
                            },
                        ),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "borderColor": "#007BFF",
                            "backgroundColor": "#f9f9f9",
                            "marginBottom": "20px",
                        },
                        multiple=False,
                    ),
                    html.Div(
                        id="file-status",  # 状态展示区域
                        style={"textAlign": "center", "marginTop": "10px"},
                    ),
                ]
            ),
            # 问题选择和分析按钮区域
            html.Div(
                [
                    dcc.Dropdown(
                        id="question-dropdown",
                        options=[
                            {"label": "Why (SHAP)", "value": "shap"},
                            {"label": "Where (Grad-CAM)", "value": "grad_cam"},
                            {"label": "What If", "value": "what_if"},
                            {"label": "How (Global SHAP)", "value": "global_shap"},
                            {"label": "Why Not (Counterfactual)", "value": "counterfactual"},
                        ],
                        placeholder="Select a question to explore",
                    ),
                    html.Div(
                        dbc.Button(
                            "Analyze",
                            id="submit-button",
                            color="primary",
                            style={
                                "marginTop": "10px",
                                "width": "100%",
                                "fontSize": "16px",
                                "padding": "10px",
                                "borderRadius": "5px",
                            },
                        ),
                        style={"textAlign": "center", "marginTop": "10px"},
                    ),
                ]
            ),
            html.Div(
                id="loading-overlay",
                children=[
                    html.Div("Analyzing...", className="loading-spinner"),
                ],
                style={
                    "display": "none",
                },
            ),
            # 文件删除按钮区域
            html.Div(
                id="remove-file-button",
                style={"marginTop": "10px", "textAlign": "center"},
            ),
            # 输出区域
            html.Div(
                id="output-container",
                style={"marginTop": "20px"},
            ),
        ],
    )
