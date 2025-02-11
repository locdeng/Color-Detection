import os
import cv2
import numpy as np


def color_histogram_of_training_image(img_name, label):
    """Extracts the most frequent color intensities and saves to training.data."""

    # Load image
    image = cv2.imread(img_name)

    if image is None:
        print(f"Error: Unable to read {img_name}")
        return

    chans = cv2.split(image)  # Split into Blue, Green, Red
    colors = ('b', 'g', 'r')
    feature_data = []

    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])  # Compute histogram
        peak_intensity = int(np.argmax(hist))  # Most frequent pixel value
        feature_data.append(str(peak_intensity))

    # Save data
    with open('training.data', 'a') as myfile:
        myfile.write(','.join(feature_data) + ',' + label + '\n')


# Define dataset path
labeled_image_path = r'C:\Users\ssuan\PycharmProjects\ComputerVision_ColorDetection\venv\training_dataset'

# Define label mapping
label_mapping = {
    'red': 'red',
    'yellow': 'yellow',
    'violet': 'violet',
    'blue': 'blue',
    'orange': 'orange',
    'green': 'green'
}

# Process each labeled folder
for label, folder_name in label_mapping.items():
    folder_path = os.path.join(labeled_image_path, folder_name)

    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Skipping.")
        continue

    for f in os.listdir(folder_path):
        img_path = os.path.join(folder_path, f)
        color_histogram_of_training_image(img_path, label)
