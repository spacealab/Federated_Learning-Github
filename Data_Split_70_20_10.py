import os
import shutil
from sklearn.model_selection import train_test_split

# Main data directory
base_dir = "client3"  # Replace this with the path to the main data folder

# Directories for images and labels
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Output directories
output_dirs = {
    "train": {
        "images": os.path.join(base_dir, "train", "images"),
        "labels": os.path.join(base_dir, "train", "labels"),
    },
    "val": {
        "images": os.path.join(base_dir, "val", "images"),
        "labels": os.path.join(base_dir, "val", "labels"),
    },
    "test": {
        "images": os.path.join(base_dir, "test", "images"),
        "labels": os.path.join(base_dir, "test", "labels"),
    },
}

# Create output directories
for split, paths in output_dirs.items():
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

# List of image and label files
image_files = sorted(os.listdir(images_dir))
label_files = sorted(os.listdir(labels_dir))

# Ensure the number of images and labels is equal
assert len(image_files) == len(label_files), "The number of images and labels does not match!"

# Split the data
train_images, test_images, train_labels, test_labels = train_test_split(
    image_files, label_files, test_size=0.3, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.33, random_state=42
)

# Function to move files to the corresponding directories
def move_files(files, source_dir, dest_dir):
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

# Move the data
move_files(train_images, images_dir, output_dirs["train"]["images"])
move_files(train_labels, labels_dir, output_dirs["train"]["labels"])
move_files(val_images, images_dir, output_dirs["val"]["images"])
move_files(val_labels, labels_dir, output_dirs["val"]["labels"])
move_files(test_images, images_dir, output_dirs["test"]["images"])
move_files(test_labels, labels_dir, output_dirs["test"]["labels"])

print("Data split completed successfully!")