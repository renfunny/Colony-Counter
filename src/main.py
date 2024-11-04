from image_processing import preprocess_image

image_path = "images/example-1.png"

# Pre-process the image
processed_image, colony_count = preprocess_image(image_path)
print(f"Colony count: {colony_count}")