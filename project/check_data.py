import os
import yaml

# Load data.yaml file
data_yaml_path = './data.yaml'

if not os.path.exists(data_yaml_path):
    print(f"Error: '{data_yaml_path}' file does not exist.")
    exit()

with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Extract paths from data.yaml
train_path = data_config.get('train', None)
val_path = data_config.get('val', None)

# Check if train and val paths exist
def check_data_paths(images_path, labels_path):
    if not os.path.exists(images_path):
        print(f"Error: Images folder does not exist at '{images_path}'")
    else:
        print(f"Images folder found at '{images_path}'")
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels folder does not exist at '{labels_path}'")
    else:
        print(f"Labels folder found at '{labels_path}'")

    # Check if there are any images and labels in the folder
    if os.path.exists(images_path) and os.listdir(images_path):
        print(f"Found {len(os.listdir(images_path))} images in '{images_path}'")
    else:
        print(f"No images found in '{images_path}'")

    if os.path.exists(labels_path) and os.listdir(labels_path):
        print(f"Found {len(os.listdir(labels_path))} label files in '{labels_path}'")
    else:
        print(f"No label files found in '{labels_path}'")

# Check train dataset
if train_path:
    train_images = os.path.join(train_path, 'images')
    train_labels = os.path.join(train_path, 'labels')
    print("\nChecking training data...")
    check_data_paths(train_images, train_labels)
else:
    print("Error: Training path not defined in data.yaml")

# Check validation dataset
if val_path:
    val_images = os.path.join(val_path, 'images')
    val_labels = os.path.join(val_path, 'labels')
    print("\nChecking validation data...")
    check_data_paths(val_images, val_labels)
else:
    print("Error: Validation path not defined in data.yaml")