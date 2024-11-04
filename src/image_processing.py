import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return None
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the agar plate's edges 
    edges = cv2.HoughCircles(
        gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=100, maxRadius=200
    )

    # Create a mask to remove the agar plate's edges
    mask = np.zeros_like(gray_image)
    if edges is not None:
        edges = np.uint16(np.around(edges))
        for x, y, r in edges[0, :]:
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Remove the agar plate's edges
    masked_gray_image = cv2.bitwise_and(gray_image, mask)

    # Minimize reflections 
    blurred_image = cv2.GaussianBlur(masked_gray_image, (5, 5), 0)

    # Apply Otsu's thresholding (Otsu's method automatically calculates the threshold value)
    thresh_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

    # Perform morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Define the minimum and maximum size of the contours
    min_size = 100
    max_size = 5000

    # Analyse contours in the image
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if min_size < cv2.contourArea(c) < max_size]

    # Display the pre-processed image
    plt.figure(figsize=(10,4))
    plt.subplot(1, 3, 1)
    plt.title("Grayscale")
    plt.imshow(gray_image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Threshold")
    plt.imshow(thresh_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Morphological Operation")
    plt.imshow(morph_image, cmap='gray')
    plt.show()

    return morph_image, len(filtered_contours)

