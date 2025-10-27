import os
import json
import shutil
import random

import pandas as pd

from tqdm import tqdm
from pathlib import Path

# === SETTINGS === #
DATASET1_FOLDER = Path('CoastlineDataset')
DATASET2_FOLDER = Path('CoastlineV2Dataset')

ID_SHIFT = 37270 - 750
VIDEO_SHIFT = 82

SPLIT_RATIO = 0.8

def copy_coastline():
     # 1. Move Images and Labels D1 -> D2
    d1_images_folder = DATASET1_FOLDER / "images"
    d1_labels_folder = DATASET1_FOLDER / "labels"
    
    d1_images_path = sorted(d1_images_folder.glob('*'))

    d2_images_folder = DATASET2_FOLDER / "images"
    d2_labels_folder = DATASET2_FOLDER / "labels"

    for frame_id, d1_image_path in tqdm(enumerate(d1_images_path), desc="Copying Images & Labels        ", total=len(d1_images_path)):

        d1_label_path = d1_labels_folder / (f"frame_{frame_id:06d}.txt")

        new_frame_name = f"frame_{(frame_id + ID_SHIFT):06d}"

        d2_image_path = d2_images_folder / (f"{new_frame_name}.jpg")
        d2_label_path = d2_labels_folder / (f"{new_frame_name}.txt")

        shutil.copy2(src=d1_image_path, dst=d2_image_path)
        shutil.copy2(src=d1_label_path, dst=d2_label_path)

    # 2. Move Updated CSV D1 -> D2
    d1_tracking_labels_folder = DATASET1_FOLDER / "tracking_labels"
    d2_tracking_labels_folder = DATASET2_FOLDER / "tracking_labels"
    
    d1_tracking_labels_path = sorted(d1_tracking_labels_folder.glob('*'))

    for video_id, tracking_labels_file in tqdm(enumerate(d1_tracking_labels_path), desc="Copying Updated Tracking Labels", total=len(d1_tracking_labels_path)):
        new_video_id = video_id + VIDEO_SHIFT
        new_tracking_labels_file = d2_tracking_labels_folder / f"cl2_video_{new_video_id:02}.csv"

        df = pd.read_csv(tracking_labels_file)

        df['frame_id'] = df['frame_id'] + ID_SHIFT

        df.to_csv(new_tracking_labels_file, index=False)

    # 3. Update D2 JSON
    d1_json_path = DATASET1_FOLDER / "sequences.json"
    d2_json_path = DATASET2_FOLDER / "sequences.json"

    with open(d1_json_path, 'r') as d1_json_file, open(d2_json_path, 'r') as d2_json_file:
        d1_data = json.load(d1_json_file)
        d2_data = json.load(d2_json_file)

        for d1_video_key, d1_image_paths in tqdm(d1_data["videos"].items(), desc="Updating JSON File             ", total=len(d1_data["videos"].keys())):
            new_video_id = int(d1_video_key[-2:]) + VIDEO_SHIFT
            d2_video_key = f"cl2_video_{new_video_id:02d}"
            

            new_paths = []

            for d1_image_path in sorted(d1_image_paths):
                frame_id = int(Path(d1_image_path).stem[-6:])
                new_frame_name = f"frame_{(frame_id + ID_SHIFT):06d}"

                d2_image_path = str(d2_images_folder / f"{new_frame_name}.jpg")

                new_paths.append(f"datasets/{d2_image_path}")

            d2_data["videos"][d2_video_key] = new_paths
        
    
    with open(d2_json_path, 'w') as f:
        json.dump(d2_data, f)

def fix_category_id():
    d2_tracking_labels_folder = DATASET2_FOLDER / "tracking_labels"
    
    d2_tracking_labels_path = sorted(d2_tracking_labels_folder.glob('*'))

    for video_id, tracking_labels_file in tqdm(enumerate(d2_tracking_labels_path), desc="Copying Updated Tracking Labels", total=len(d2_tracking_labels_path)):
        if video_id < 82:

            df = pd.read_csv(tracking_labels_file)

            df['category_id'] = df['category_id'] + 1

            df.to_csv(tracking_labels_file, index=False)

if __name__ == "__main__":
    
    copy_coastline()

    # d2_images_folder = DATASET2_FOLDER / "images"

    # images_path_list = sorted(d2_images_folder.glob('*'))

    # train_val_split = images_path_list[:ID_SHIFT]

    # test_split = images_path_list[ID_SHIFT:]

    # random.shuffle(train_val_split)

    # n_train = int(ID_SHIFT * SPLIT_RATIO)

    # train_split = train_val_split[:n_train]
    # val_split   = train_val_split[n_train:]
    # test_split  = images_path_list[ID_SHIFT:]

    # # === WRITE TXT FILES ===
    # def write_paths(image_list, output_path):
    #     with open(output_path, "w") as f:
    #         for img in image_list:
    #             f.write(f"datasets/{str(img)}\n")

    # write_paths(sorted(train_split), DATASET2_FOLDER / 'train_txt')
    # write_paths(sorted(val_split), DATASET2_FOLDER / 'val_txt')
    # write_paths(sorted(test_split), DATASET2_FOLDER / 'test_txt')


    # print(len(train_split), len(val_split))
