from ultralytics import YOLO
import os

# Load the trained YOLO model
model = YOLO('./runs/detect/train6/weights/best.pt')

# Path to the directory containing test images
test_images_dir = "./datasets/datasets/val/images/"
output_dir = "./runs/detect/batch_test/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Perform inference on each image in the directory
for image_file in os.listdir(test_images_dir):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
        image_path = os.path.join(test_images_dir, image_file)
        print(f"Processing: {image_path}")
        
        # Perform inference and save results
        results = model.predict(source=image_path, save=True, project=output_dir)
        print(f"Saved results for {image_file} to {output_dir}")