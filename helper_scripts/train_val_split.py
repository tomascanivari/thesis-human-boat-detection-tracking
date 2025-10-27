import os
import random
import shutil
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    # List all image files
    image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])

    # Shuffle and split
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Create subdirectories
    for subdir in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subdir), exist_ok=True)

    def move_files(file_list, target_subdir):
        print(f"Processing {target_subdir} files:")
        for image_file in tqdm(file_list, desc=f"{target_subdir.capitalize()}"):
            base_name = os.path.splitext(image_file)[0]
            label_file = base_name + ".txt"

            img_src = os.path.join(images_dir, image_file)
            lbl_src = os.path.join(labels_dir, label_file)
            img_dst = os.path.join(images_dir, target_subdir, image_file)
            lbl_dst = os.path.join(labels_dir, target_subdir, label_file)

            if os.path.exists(lbl_src):
                shutil.move(img_src, img_dst)
                shutil.move(lbl_src, lbl_dst)
            else:
                print(f"Warning: No matching label for image {image_file}, skipping.")

    move_files(train_files, 'train')
    move_files(val_files, 'val')

# Example usage
images_folder = "images"
labels_folder = "labels"
split_dataset(images_folder, labels_folder)