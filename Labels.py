import os
import json
from pycocotools.coco import COCO

# File paths
annotations_path = "annotations_trainval2017/annotations/instances_train2017.json"  # Path to the JSON annotations file
images_dir = "train2017/"  # Path to the original images directory
output_dir = "human_data/"  # Directory to store extracted human images
labels_dir = "labels/"  # Directory to store YOLO format labels

# Create directories for extracted images and labels
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Load the JSON file
coco = COCO(annotations_path)

# Get the category ID for "person"
category_id = coco.getCatIds(catNms=["person"])
image_ids = coco.getImgIds(catIds=category_id)

# Process images and labels
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    width = img_info["width"]
    height = img_info["height"]

    # Copy the image to the new directory
    src_path = os.path.join(images_dir, file_name)
    dst_path = os.path.join(output_dir, file_name)
    if os.path.exists(src_path):
        os.rename(src_path, dst_path)

    # Extract bounding box information for each image
    annotation_ids = coco.getAnnIds(imgIds=img_id, catIds=category_id, iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)

    # Create a label file for each image
    label_path = os.path.join(labels_dir, file_name.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        for ann in annotations:
            bbox = ann["bbox"]
            x, y, w, h = bbox
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height
            # Write data in YOLO format: <class> <x_center> <y_center> <width> <height>
            f.write(f"0 {x_center} {y_center} {w} {h}\n")

print("Labels and images prepared successfully.")