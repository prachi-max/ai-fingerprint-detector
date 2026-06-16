import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224

# Load model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
    interpreter.invoke()

    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    confidence = round(prob * 100, 2)

    if prob > 0.5:
        return "🟢 LIVE FINGERPRINT", confidence
    else:
        return "🔴 SPOOF FINGERPRINT", confidence


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    #  AI Fingerprint Detection
    
    Upload a fingerprint image below 
    """)

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Fingerprint")
        output_label = gr.Textbox(label="Prediction")

    confidence_bar = gr.Slider(
        minimum=0,
        maximum=100,
        label="Confidence (%)",
        interactive=False
    )

    analyze_btn = gr.Button("Analyze Fingerprint", variant="primary")

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[output_label, confidence_bar]
    )

    gr.Markdown("""
    ---
    """)

demo.launch()