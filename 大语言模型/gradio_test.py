import gradio as gr
import cv2
import numpy as np
import os


def detect_edges(image):
    if image is None:
        return None

    # 转换颜色空间 RGB to BGR (OpenCV默认使用BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 边缘检测
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 转换回RGB格式返回给Gradio
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


# 使用相对路径指向示例图片
example_image = "WechatIMG669.jpg"  # 确保图片在相同目录下

# 检查图片是否存在
if not os.path.exists(example_image):
    print(f"警告: 示例图片 {example_image} 不存在!")
    example_image = None

with gr.Blocks() as demo:
    gr.Markdown("## 图像边缘检测演示")

    with gr.Row():
        input_image = gr.Image(label="上传图片", type="numpy")
        output_image = gr.Image(label="检测结果")

    detect_btn = gr.Button("开始检测")
    detect_btn.click(detect_edges, inputs=input_image, outputs=output_image)

    # 示例部分（仅在图片存在时添加）
    if example_image:
        gr.Examples(
            examples=[[example_image]],
            inputs=input_image,
            outputs=output_image,
            fn=detect_edges,
            cache_examples=False
        )

# 启动参数
demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)