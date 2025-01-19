import os
import shutil

# Paths
images_dir = './clientـmain//images/'
labels_dir = './clientـmain//labels/'
output_dir = './dataset'

# Create dataset structure
os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/labels'), exist_ok=True)

# Split data into train and validation sets (80% train, 20% validation)
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
split_idx = int(len(image_files) * 0.8)

train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Copy files to train and validation folders
for image_file in train_files:
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, 'train/images', image_file))
    label_file = image_file.replace('.jpg', '.txt')
    shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, 'train/labels', label_file))

for image_file in val_files:
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, 'val/images', image_file))
    label_file = image_file.replace('.jpg', '.txt')
    shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, 'val/labels', label_file))

print("Data preparation completed.")