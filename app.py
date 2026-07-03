"""Gradio demo for pig detection and counting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image

from src.inference import load_yolo_model, run_inference
from src.utils import load_config

DEFAULT_WEIGHTS = "outputs/pig_yolov8n/weights/best.pt"


def predict_image(
    image: Image.Image | None,
    weights_path: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
) -> tuple[Any, str, str, str, str, pd.DataFrame | str]:
    """Gradio prediction function returning image, metrics, and detection table."""
    if image is None:
        return None, "", "", "", "请上传猪舍图片。", pd.DataFrame()

    model, message = load_yolo_model(weights_path)
    if model is None:
        return (
            None,
            "",
            "",
            "",
            f"模型权重未找到：{message}，请先训练模型或提供有效权重路径。",
            pd.DataFrame(),
        )

    result = run_inference(model, image, conf=conf)
    annotated_image = result.pop("annotated_image")

    detections = result["detections"]
    if detections:
        df = pd.DataFrame(detections)
        df = df[["class_name", "confidence", "x1", "y1", "x2", "y2", "area_ratio"]]
    else:
        df = pd.DataFrame(columns=["class_name", "confidence", "x1", "y1", "x2", "y2", "area_ratio"])

    return (
        annotated_image,
        str(result["pig_count"]),
        f"{result['mean_confidence']:.4f}",
        f"{result['bbox_area_ratio']:.4f}",
        result["risk_reason"],
        df,
    )


def build_demo() -> gr.Interface:
    cfg = load_config()
    default_weights = DEFAULT_WEIGHTS

    with gr.Blocks(title="猪只检测与计数分析系统") as demo:
        gr.Markdown("# 猪只检测与计数分析系统")
        gr.Markdown("上传猪舍图片，系统会检测猪只、统计数量、分析密度并提示拥挤风险。")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传猪舍图片")
                weights_path = gr.Textbox(value=default_weights, label="模型权重路径")
                conf_slider = gr.Slider(minimum=0.05, maximum=1.0, value=0.25, step=0.05, label="置信度阈值")
                submit_btn = gr.Button("开始检测")

            with gr.Column():
                output_image = gr.Image(type="pil", label="检测结果")
                pig_count = gr.Textbox(label="猪只数量")
                mean_conf = gr.Textbox(label="平均置信度")
                area_ratio = gr.Textbox(label="检测框面积占比")
                risk_reason = gr.Textbox(label="风险原因")
                detections_table = gr.Dataframe(label="检测框详情")

        submit_btn.click(
            fn=predict_image,
            inputs=[input_image, weights_path, conf_slider],
            outputs=[output_image, pig_count, mean_conf, area_ratio, risk_reason, detections_table],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
