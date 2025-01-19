from pycocotools.coco import COCO
import os
import shutil

# Paths
annotations_path = "annotations_trainval2017/annotations/instances_train2017.json"
images_dir = "train2017/"
output_dir = "human_data/"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load annotations
coco = COCO(annotations_path)

# Get IDs for "person" category
category_id = coco.getCatIds(catNms=["person"])
image_ids = coco.getImgIds(catIds=category_id)

# Copy images to output directory
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    src_path = os.path.join(images_dir, img_info["file_name"])
    dst_path = os.path.join(output_dir, img_info["file_name"])
    shutil.copy(src_path, dst_path)
    print(f"Copied {img_info['file_name']} to {output_dir}")