# Color Detection using OpenCV and KNN Classifier

## Overview
This project is a **color detection system** that utilizes **OpenCV** for real-time color recognition through a webcam and a **K-Nearest Neighbors (KNN) classifier** to classify colors based on a training dataset. The program processes video frames, detects colors, and labels them accordingly.

## Features
- Real-time color detection using OpenCV.
- Histogram-based color classification.
- Uses KNN algorithm to recognize colors.
- Predefined color ranges for **red, orange, yellow, green, blue, and violet**.
- Saves training data for classification.

### How It Works
1. The program checks if the **training dataset** exists.
2. The webcam captures live frames.
3. The frame is converted to HSV for color detection.
4. Predefined color ranges are applied to filter and detect colors.
5. Bounding boxes and labels are displayed for detected colors.
6. The **KNN classifier** predicts the color from trained data.

## Project Structure
```
├── color_api/
│   ├── color_histogram_test_image.py  # Extracts color histograms from test images
│   ├── color_histogram_training_image.py  # Extracts color histograms from training images
│   ├── knn.py  # K-Nearest Neighbors classifier for color classification
├── main.py  # Main script for real-time color detection
├── requirements.txt  # Dependencies
├── training.data  # Stored color histogram training data
└── test.data  # Color data for classification
```

## Code Explanation
- `main.py` initializes the webcam, applies color filtering, and detects colors.
- `color_histogram_training_image.py` processes training images and extracts color histograms.
- `color_histogram_test_image.py` extracts color histograms from test images.
- `knn.py` implements the KNN algorithm to classify colors based on training data.

## Notes
- Ensure `training.data` is generated before running the program.
- The dataset should be updated if more colors need to be classified.

## Acknowledgments
This project is built using **OpenCV**, **NumPy**, and **Scikit-Learn** for image processing and classification.

## Result
![Image](https://github.com/user-attachments/assets/7577b8ee-9b74-4467-ae98-9c2b87ac4ac7)
![Image](https://github.com/user-attachments/assets/xv-1dbf-40fd-a8c5-30ead11de4f9)
![Image](https://github.com/user-attachments/assets/493d0ef4-2a95-4823-8ece-7e1874d59c29)

