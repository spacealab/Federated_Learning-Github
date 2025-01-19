import os
import shutil

# Paths for image and label directories
images_dir = "human_data"
labels_dir = "labels"

# Paths for new directories for each client
clients = ["client1", "client2", "client3"]
for client in clients:
    os.makedirs(os.path.join(client, "images"), exist_ok=True)
    os.makedirs(os.path.join(client, "labels"), exist_ok=True)

# Get the list of files
image_files = sorted(os.listdir(images_dir))
label_files = sorted(os.listdir(labels_dir))

# Ensure the number of images and labels are the same
assert len(image_files) == len(label_files), "Number of images and labels must be the same."

# Divide the files into three chunks
chunk_size = len(image_files) // len(clients)

for i, client in enumerate(clients):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size if i < len(clients) - 1 else len(image_files)

    for j in range(start_idx, end_idx):
        # Source paths for files
        img_src = os.path.join(images_dir, image_files[j])
        lbl_src = os.path.join(labels_dir, label_files[j])

        # Destination paths for files
        img_dst = os.path.join(client, "images", image_files[j])
        lbl_dst = os.path.join(client, "labels", label_files[j])

        # Copy files to the destination directory
        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

print("Images and labels have been successfully distributed among clients.")