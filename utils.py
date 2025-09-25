import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(uploaded_file, img_size=(128,128)):
    """Preprocess uploaded image for model prediction"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_resized = img_resized / 255.0
    return np.expand_dims(img_resized, axis=0), img

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

    # Draw path between diseased spots
    xs, ys = zip(*diseased_spots)
    plt.plot(ys, xs, "r--", linewidth=2, marker="o")

    plt.title(" Drone Spraying Path Simulation")
    return plt
