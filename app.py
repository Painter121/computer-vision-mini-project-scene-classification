import os
import gradio as gr
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

model = tf.keras.models.load_model("scene_model.keras")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# กลับ dict จาก {class: index} → {index: class}
index_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img):
    if img is None:
        return None

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)  # สำคัญมาก
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    results = {
        index_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return results

sample_dir = "samples"
examples = [
    [os.path.join(sample_dir, f)]
    for f in sorted(os.listdir(sample_dir))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
] if os.path.exists(sample_dir) else None

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="อัปโหลดรูปภาพ"),
    outputs=gr.Label(num_top_classes=3, label="ผลการทำนาย"),
    title="Scene Classification",
    description="อัปโหลดรูปภาพเพื่อทำนายฉาก (Scene) หรือคลิกเลือกภาพตัวอย่างด้านล่าง",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    interface.launch()
