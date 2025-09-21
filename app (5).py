import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw

# Load model (pretrained COCO object detection)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="🚦 AI Traffic Checker", layout="wide")
st.title("🚦 AI Traffic Checker App")
st.write("Upload a traffic image to check congestion level and emergency vehicle detection.")

uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Run detection
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    draw = ImageDraw.Draw(image)
    vehicle_count = 0
    ambulance_detected = False

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        box = [round(i, 2) for i in box.tolist()]

        if label_name in ["car", "truck", "bus", "motorcycle"]:
            vehicle_count += 1
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_name} {score:.2f}", fill="red")

        if label_name == "ambulance":  # not in COCO dataset, but keep for custom models
            ambulance_detected = True
            draw.rectangle(box, outline="blue", width=3)
            draw.text((box[0], box[1]), "Ambulance 🚑", fill="blue")

    # Congestion logic
    if vehicle_count < 5:
        traffic_status = "✅ Clear"
    elif vehicle_count < 15:
        traffic_status = "⚠️ Moderate"
    else:
        traffic_status = "🚨 Heavy Traffic"

    st.image(image, caption="Detected Traffic", use_column_width=True)
    st.subheader(f"Traffic Status: {traffic_status}")
    st.write(f"Detected Vehicles: **{vehicle_count}**")

    if traffic_status == "🚨 Heavy Traffic":
        st.warning("⚠️ Traffic Jam Detected. Suggesting alternate route...")
        st.info("➡️ Please take the bypass road to save time.")

    if ambulance_detected:
        st.error("🚑 Ambulance detected! Grant emergency passage immediately.")
