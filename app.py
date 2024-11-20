import gradio as gr
import numpy as np
from PIL import Image
import cv2

def image_modifier(img, brightness, contrast):
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    brightness_matrix = np.ones(img.shape, dtype="uint8") * brightness
    brightened_img = cv2.add(img, brightness_matrix)
    
    contrast_matrix = np.ones(brightened_img.shape) * contrast
    contrasted_img = cv2.multiply(brightened_img, contrast_matrix)
    
    final_img = np.clip(contrasted_img, 0, 255).astype(np.uint8)
    
    return final_img

def text_analyzer(text):
    num_words = len(text.split())
    num_chars = len(text)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "Word Count": num_words,
        "Character Count": num_chars,
        "Sentence Count": num_sentences
    }

with gr.Blocks(title="Multi-Tool Demo App") as demo:
    gr.Markdown("# Welcome to the Multi-Tool Demo App")
    
    with gr.Tab("Image Modifier"):
        gr.Markdown("## Image Modification Tool")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image")
                brightness_slider = gr.Slider(minimum=-50, maximum=50, value=0, 
                                           step=1, label="Brightness")
                contrast_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, 
                                          step=0.1, label="Contrast")
                image_button = gr.Button("Process Image")
            with gr.Column():
                image_output = gr.Image(label="Processed Image")
    
    with gr.Tab("Text Analyzer"):
        gr.Markdown("## Text Analysis Tool")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Enter Text", lines=5)
                text_button = gr.Button("Analyze Text")
            with gr.Column():
                text_output = gr.JSON(label="Analysis Results")
    
    image_button.click(
        fn=image_modifier,
        inputs=[image_input, brightness_slider, contrast_slider],
        outputs=image_output
    )
    
    text_button.click(
        fn=text_analyzer,
        inputs=text_input,
        outputs=text_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)