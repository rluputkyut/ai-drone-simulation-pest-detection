import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "plant_disease_model_multiclass.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load class labels
# -----------------------------
# Safer: load from JSON if you saved earlier
if os.path.exists("class_labels.json"):
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
else:
    # fallback: read directly from dataset (train folder)
    DATASET_PATH = "dataset/train"
    class_labels = sorted(os.listdir(DATASET_PATH))

# -----------------------------
# Helper: Preprocess image
# -----------------------------
def preprocess_image(uploaded_file, img_size=(128,128)):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_resized = img_resized / 255.0
    return np.expand_dims(img_resized, axis=0), img

# -----------------------------
# Helper: Drone Simulation
# -----------------------------
def simulate_drone_field(field_size=10, diseased_spots=None):
    """Simulate drone path visiting diseased spots"""
    field = np.zeros((field_size, field_size))
    if diseased_spots is None:
        diseased_spots = [(2,3), (5,6), (7,8)]
    for x,y in diseased_spots:
        field[x,y] = 1

    plt.imshow(field, cmap="Greens")
    for (x,y) in diseased_spots:
        plt.text(y, x, "X", ha="center", va="center", color="red", fontsize=14)

    # Connect diseased spots with path
    xs, ys = zip(*diseased_spots)
    plt.plot(ys, xs, "r--", linewidth=2, marker="o")

    plt.title("🛩 Drone Spraying Path Simulation")
    return plt

# -----------------------------
# Streamlit UI
# -----------------------------
st.title(" AI-Powered Prototype for Plant Monitoring")
st.write("Upload crop images to detect diseases and simulate targeted pesticide spraying.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

diseased_count = 0
total_count = 0
diseased_spots = []

if uploaded_files:
    st.subheader("Detection Results")
    for i, uploaded_file in enumerate(uploaded_files):
        img_array, img_display = preprocess_image(uploaded_file)

        # Run prediction
        pred = model.predict(img_array)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))

        total_count += 1

        # Decide healthy vs diseased
        predicted_label = class_labels[class_id]
        if "healthy" not in predicted_label.lower():
            diseased_count += 1
            diseased_spots.append((np.random.randint(0,10), np.random.randint(0,10)))  # random coords

        # Show result
        # st.image(img_display, caption=f"Image {i+1}: {predicted_label} ({confidence:.2f})", use_column_width=True)
        st.image(img_display, caption=f"{predicted_label}", width=150)
        with st.expander("See full image"):
            st.image(img_display, use_column_width=True)

    # Sustainability impact
    if total_count > 0:
        reduction = ((total_count - diseased_count) / total_count) * 100
        st.subheader(" Sustainability Impact")
        st.write(f"Total Plants Scanned: **{total_count}**")
        st.write(f"Diseased Detected: **{diseased_count}**")
        st.write(f"Estimated Pesticide Reduction: **{reduction:.1f}%**")

    # Drone simulation
    # if diseased_spots:
    #     st.subheader(" Drone Spraying Simulation")
    #     fig = simulate_drone_field(10, diseased_spots)
    #     st.pyplot(fig)
