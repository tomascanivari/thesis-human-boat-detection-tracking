import os
import re
import cv2
import csv
import json
import shutil
import zipfile
import subprocess

import pandas as pd

from glob import glob
from tqdm import tqdm
from hashlib import md5
from pathlib import Path

##############################
# ==== GENERAL SETTINGS ==== #
##############################

# == CVAT CONFIGURATION == #
AUTH              = "tomascrispim2@hotmail.com:teste123CVAT"
SERVER_HOST       = "http://localhost:8080"
ANNOTATION_FORMAT = "Ultralytics YOLO Detection Track 1.0"

# == FOLDERS & PATHS CONFIGURATION == #
DATASET_FOLDER                 = Path("CoastlineV2Dataset")
DATASET_IMAGES_FOLDER          = DATASET_FOLDER / "images"
DATASET_LABELS_FOLDER          = DATASET_FOLDER / "labels"
DATASET_TRACKING_LABELS_FOLDER = DATASET_FOLDER / "tracking_labels"
TEMP_DOWNLOAD_FOLDER           = DATASET_FOLDER / "temp_download"
SEQUENCES_JSON_PATH            = DATASET_FOLDER / "sequences.json"

os.makedirs(name=TEMP_DOWNLOAD_FOLDER, exist_ok=True)

# == IMAGES & LABELS CONFIGURATION == #
IMG_SIZE     = (640, 485)
FRAME_RATE   = 30
ORIGINAL_EXT = '.png'
RESIZED_EXT  = '.jpg'
CSV_HEADER   = ['frame_id', 'track_id', 'category_id', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']

# DATASETS SPLITS
TRAIN_VIDEO_KEY = set([f"cl2_video_{id:02d}" for id in range(16, 82)])
VAL_VIDEO_KEY   = set([f"cl2_video_{id:02d}" for id in range(0, 16)])
TEST_VIDEO_KEY  = set([f"cl2_video_{id:02d}" for id in range(82, 94)])

##############################
# ==== HELPER FUNCTIONS ==== #
##############################

def get_completed_tasks_set():

    # 1. Run the CLI command and get JSON output
    result = subprocess.run(
        ["cvat-cli", "--server-host", SERVER_HOST, "--auth", AUTH, "task", "ls", "--json"],
        capture_output=True,
        text=True,
        check=True
    )

    # 2. Parse the JSON
    tasks = json.loads(result.stdout)

    # 3. Filter tasks with completed jobs
    completed_task_ids = sorted([
        f"{task['id']:02}"
        for task in tasks
        if isinstance(task.get("jobs"), dict) and task["jobs"].get("completed", 0) > 0
    ])

    return set(completed_task_ids)

def get_processed_tasks_set(sequences_data):
    return set([video_name[-2:] for video_name in sequences_data["videos"]])

def download_task(task_id, task_folder):
    
    # 1. Create '.zip' path
    zip_path         = TEMP_DOWNLOAD_FOLDER / f"task_{task_id}.zip"

    # 2. Run the CLI command 
    print(f"  Downloading Task {task_id} to {task_folder}.")
    result = subprocess.run(["cvat-cli", "--server-host", SERVER_HOST, "--auth", AUTH, "task", "export-dataset", task_id, zip_path, "--format", ANNOTATION_FORMAT, "--with-images", "yes", "--completion_verification_period", "300"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    if result.returncode != 0:
        print(f"  Failed to download task {task_id}:", result.stderr)
        return
    
    # 3. Extract the '.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(task_folder)
        os.remove(zip_path)

    # 4. Clear the DOCKER EXPORT CACHE
    subprocess.run("docker exec cvat_server bash -c 'rm -rf /home/django/data/cache/export/*'", shell=True, check=True)

def resize_images(images_folder):

    # 1. Process Each Image
    for image_path in tqdm(sorted(glob(os.path.join(images_folder, f"*{ORIGINAL_EXT}"))), desc="  Resizing", leave=True):
                                                                                                
        # 1.1 Load & Resize Image
        img = cv2.resize(cv2.imread(image_path), IMG_SIZE)                

        # 1.2 Save Resized Image
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(f"{images_folder}/{img_name}{RESIZED_EXT}", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 1.3 Delete Original Image
        os.remove(image_path)

def sync_images_and_labels(images_folder, labels_folder):

    for filename in os.listdir(images_folder):
        if filename.lower().endswith(RESIZED_EXT):
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(labels_folder, base_name + '.txt')
            if not os.path.exists(txt_path):
                # Create empty file
                with open(txt_path, 'w') as f:
                    pass

    print(f"  Synced Images with Labels [{len(os.listdir(images_folder))}/{len(os.listdir(images_folder))}].")

def remove_duplicates(images_folder, labels_folder):
   
    # 1. Create Variables
    hash_map = {}
    unique_label_paths = []

    # 2. Process Labels to Find Duplicates
    label_files = sorted(labels_folder.glob('frame_*.txt'))

    for label_path in label_files:
        if label_path.stat().st_size == 0:
            # Blank file: keep unconditionally
            unique_label_paths.append(label_path)
        else:
            with open(label_path, 'rb') as f:
                label_content = f.read()
                
            label_hash = md5(label_content).hexdigest()
            # Overwrite with last occurrence of this content
            hash_map[label_hash] = label_path

    # 2.1 Add non-blank, deduplicated ones
    unique_label_paths.extend(hash_map.values())

    # 2.3 Remove all Duplicates
    label_paths_set = set(label_files)
    unique_label_paths_set = set(unique_label_paths)

    removed_label_paths_set = set(label_paths_set - unique_label_paths_set)

    for rm_label_path in removed_label_paths_set:
        base_name = rm_label_path.stem
        rm_image_path = images_folder / (base_name + RESIZED_EXT)
        
        os.remove(rm_label_path)
        os.remove(rm_image_path)

    # 2.2 Optional sort to ensure consistent naming
    unique_label_paths = sorted(set(unique_label_paths))

    # 3. Rename All the Unique Labels & Images 
    for new_idx, label_path in enumerate(unique_label_paths):
        base_name = label_path.stem
        image_path = images_folder / (base_name + RESIZED_EXT)

        # 3.1 Rename Label and Image
        new_filename = f"temp_frame_{new_idx:06d}"
        new_label_path = labels_folder / (new_filename + '.txt')
        new_image_path = images_folder / (new_filename + '.jpg')

        # 3.2 Copy New Label and Image
        os.rename(label_path, new_label_path)
        os.rename(image_path, new_image_path)

    total_labels = len(label_files)
    kept_labels = len(unique_label_paths)
    removed_labels = total_labels - kept_labels

    print(f"  Removed {removed_labels} duplicates. Kept {kept_labels} Images & Labels.")

def move_to_dataset(video_name, images_folder, labels_folder):

    # 1. Read All Files Currently on Dataset Folder  
    dataset_labels_list = glob(f"{DATASET_LABELS_FOLDER}/frame_*.txt")

    if len(dataset_labels_list) == 0:
        max_num = 0
    else:
        max_num = max(int(re.search(r'frame_(\d+)', os.path.basename(f)).group(1)) for f in dataset_labels_list) + 1
    
    # 2. Create the CSV File 
    csv_path = DATASET_TRACKING_LABELS_FOLDER / f"{video_name}.csv"
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    
    # 3. Move Images and Labels to Dataset Folder
    new_image_paths = []
    task_labels_list = sorted(labels_folder.glob('temp_frame_*.txt'))
    
    for idx, label_path in enumerate(task_labels_list):
        new_idx = max_num + idx
        base_name = label_path.stem
        image_path = images_folder / (base_name + RESIZED_EXT)

        # 3.1 Copy Label and Image
        new_filename = f"frame_{new_idx:06d}"
        new_label_path = DATASET_LABELS_FOLDER / (new_filename + '.txt')
        new_image_path = DATASET_IMAGES_FOLDER / (new_filename + RESIZED_EXT)
        
        shutil.copy2(image_path, new_image_path)

        # 3.1.1 Remove 'tracking_id' from Label File
        with open(label_path, 'r') as infile, open(new_label_path, 'w') as label_file:
            for line in infile:
                columns = line.strip().split()
                new_line = ' '.join(columns[:-1])  # Remove last column ('tracking_id')
                label_file.write(new_line + '\n')
        
                # 3.1.2 Fill Tracking CSV File
            
                # Parse input columns
                if len(columns) != 6:
                    continue  # skip malformed lines

                category_id, x, y, w, h, track_id = columns

                # Convert YOLO to MOT format
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
                x_top_left = (x - w / 2) * IMG_SIZE[0]
                y_top_left = (y - h / 2) * IMG_SIZE[1]
                w_abs = w * IMG_SIZE[0]
                h_abs = h * IMG_SIZE[1]

                # Build full row for output
                row = [
                    new_idx,            # frame_id
                    track_id,
                    category_id,
                    x_top_left, 
                    y_top_left, 
                    w_abs, 
                    h_abs,
                    1,                   # conf
                    -1, -1, -1           # xx, yy, zz
                ]

                writer.writerow(row)

        # 3.2 Append 'new_image_path'
        new_image_paths.append(f"datasets/{str(new_image_path)}")

    csv_file.close()

    return new_image_paths

def process_tracking_csv(tracking_label_path):

    # 1. Variables
    n_frames  = set()
    instances = [0, 0]

    # 2. Read CSV File
    with open(tracking_label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)  

        # 2.1 Read Each Line & Update Variables
        for row in reader:
            frame_id    = int(row['frame_id'])
            category_id = int(row['category_id'])

            n_frames.add(frame_id)      
            instances[category_id-1] += 1 

    return len(n_frames), sum(instances), instances[0], instances[1]

def update_json_and_csv():
    
    # 0. Create Dictionary for the Reordered JSON
    new_sequences_data = {"videos": {}}

    # 1. Load JSON
    with open(SEQUENCES_JSON_PATH, 'r') as f:
        sequences_data = json.load(f)

    # 1.1 Obtain All Video Names (keys)
    video_keys = sorted(sequences_data["videos"].keys())

    # 2. Reorder Images and Labels to a Temporary Folder
    old_images_folder = DATASET_FOLDER / "images"
    old_labels_folder = DATASET_FOLDER / "labels"
    
    new_images_folder = DATASET_FOLDER / "new_images"
    new_labels_folder = DATASET_FOLDER / "new_labels"

    os.makedirs(new_images_folder, exist_ok=True)
    os.makedirs(new_labels_folder, exist_ok=True)

    # 2.1 Process Each Video by Ascending Order
    index = 0
    for video_key in tqdm(video_keys, desc="Updating JSON and CSV Files", leave=True):
        
        video_id = int(video_key[-2:])

        if video_id <= 81:
        
            # 2.2 Process Each Image & Label Path
            n_empty = 0
            stop_counting = False
            new_start_frame_id = index
            new_paths = []
            for old_image_path in sequences_data["videos"][video_key]:
                    
                    # 2.2.1 Old Paths
                    old_frame_name = Path(old_image_path).stem
                    old_image_path = old_images_folder / (old_frame_name + ".jpg")
                    old_label_path = old_labels_folder / (old_frame_name + ".txt")

                    # Account For Empty Annotations in the Beginning
                    if not stop_counting:
                        if old_label_path.stat().st_size == 0:
                            n_empty += 1
                        else:
                            stop_counting = True

                    # 2.2.2 New Paths
                    new_frame_name = f"frame_{index:06d}"
                    new_image_path = new_images_folder / (new_frame_name + ".jpg")
                    new_label_path = new_labels_folder / (new_frame_name + ".txt")

                    # 2.2.3 Copy Image & Label
                    shutil.copy2(src=old_image_path, dst=new_image_path)
                    shutil.copy2(src=old_label_path, dst=new_label_path)

                    # 2.2.4 Update New Paths
                    new_image_path = old_images_folder / (new_frame_name + ".jpg") # Accounting for the folder rename
                    new_paths.append(f"datasets/{str(new_image_path)}")

                    # 2.2.5 Update Index
                    index += 1

            # 2.3 Update New JSON
            new_sequences_data["videos"][video_key] = new_paths

        # 2.4 Update CSV Tracking Labels
        df = pd.read_csv(DATASET_TRACKING_LABELS_FOLDER / (video_key + ".csv"))

        old_start_frame_id = 0 if df.empty else df['frame_id'].iloc[0]

        id_shift = new_start_frame_id - old_start_frame_id + n_empty
        df['frame_id'] = df['frame_id'] + id_shift

        df.to_csv(DATASET_TRACKING_LABELS_FOLDER / (video_key + ".csv"), index=False)

    # 3. Save Updated JSON
    for video_key, image_paths in new_sequences_data.items():   # Keep 82-93
        sequences_data[video_key] = image_paths

    with open(SEQUENCES_JSON_PATH, 'w') as f:
        json.dump(sequences_data, f)

    # 4. Rename Temporary Folder & Remove Old Folders
    shutil.rmtree(old_images_folder)
    shutil.rmtree(old_labels_folder)

    new_images_folder.rename(old_images_folder)
    new_labels_folder.rename(old_labels_folder)

def create_train_val_test_splits():
    
    train_split = []
    val_split   = []
    test_split  = []

    # 1. Load JSON
    with open(SEQUENCES_JSON_PATH, 'r') as f:
        sequences_data = json.load(f)

    # 2. Read Image Paths By Split
    for video_key, image_paths in sequences_data["videos"].items():
        if video_key in TRAIN_VIDEO_KEY:
            train_split.extend(image_paths)
        elif video_key in VAL_VIDEO_KEY:
            val_split.extend(image_paths)
        elif video_key in TEST_VIDEO_KEY:
            test_split.extend(image_paths)

    # 3. Write To '.txt' Files
    def write_paths(image_list, output_path):
        with open(output_path, "w") as f:
            for img in image_list:
                f.write(f"datasets/{str(img)}\n")

    write_paths(sorted(train_split), DATASET_FOLDER / 'train_txt')
    write_paths(sorted(val_split), DATASET_FOLDER / 'val_txt')
    write_paths(sorted(test_split), DATASET_FOLDER / 'test_txt')

#########################
# ==== MAIN SCRIPT ==== #
#########################

if __name__ == '__main__':

    # 1. Get Completed Tasks Set
    completed_tasks_set = get_completed_tasks_set()

    # 2. Load 'sequences.json' & Get Processed Tasks Set
    with open(SEQUENCES_JSON_PATH, 'r') as f:
        sequences_data = json.load(f)

    processed_tasks_set = get_processed_tasks_set(sequences_data)

    # 3. Obtain Tasks Processed and to be Processed
    already_processed_tasks_list = sorted(list(completed_tasks_set & processed_tasks_set))
    to_be_processed_tasks_list   = sorted(list(completed_tasks_set - processed_tasks_set))

    # 3.1 Print Tasks already Processed
    for task_id in already_processed_tasks_list:
        print(f"Task {task_id} already processed. Skipping...")

    print("")

    # 4. Process Each Task Left
    for task_id in to_be_processed_tasks_list:

        print(f"Processing Task {task_id}:")

        # 4.1 Create Task Folder
        task_folder = TEMP_DOWNLOAD_FOLDER / f"task_{task_id}"
        os.makedirs(name=task_folder, exist_ok=True)

        # 4.2 Define Images and Labels Folder
        images_folder = task_folder / "images/train"
        labels_folder = task_folder / "labels/train"

        os.makedirs(name=labels_folder, exist_ok=True)    # In case there are no annotation for the task

        # 4.3 Download Task
        download_task(task_id, task_folder)

        # 4.4 Resize Images to IMG_SIZE and '.jpg' extension
        resize_images(images_folder)

        # 4.5 Sync Images and Labels
        sync_images_and_labels(images_folder, labels_folder)

        # 4.6 Remove Duplicate Images & Re-order
        remove_duplicates(images_folder, labels_folder)

        # 4.7 Move Images and Labels to Dataset Folder & Fill JSON & CSV
        video_name = f"cl2_video_{task_id}"
        sequences_data["videos"][video_name] = move_to_dataset(video_name, images_folder, labels_folder)

        shutil.rmtree(task_folder)  # Delete Task Folder

        # 4.8. Save JSON with Processed Info
        f = open(SEQUENCES_JSON_PATH, "w")
        json.dump(sequences_data, f)
        f.close()

        print("")
    
    # 5. Remove temporary folder
    shutil.rmtree(TEMP_DOWNLOAD_FOLDER)

    # 6. Update JSON and CSV (Comment After Extracting All V2 videos and copying V1 to V2)
    # update_json_and_csv()
    
    # print("")

    # 7. Create Train, Val & Test Splits
    create_train_val_test_splits()

    # 7. Dataset Analysis
    dataset_tracking_labels_list = sorted(glob(f"{DATASET_TRACKING_LABELS_FOLDER}/*.csv"))

    data = {
        "Split": ["Train", "Validation", "Test", "Total"],
        "Num. Videos": [0, 0, 0, 0],
        "Num. Frames": [0, 0, 0, 0],
        "Num. Instances": [0, 0, 0, 0],
        "Num. Swimmer": [0, 0, 0, 0],
        "Num. Boat": [0, 0, 0, 0],
    }

    # 7.1 Create the DataFrame
    df = pd.DataFrame(data)

    # 7.2 Read Instances Information From CSV Files and Fill DF
    instances = [0, 0]
    for tracking_label_path in dataset_tracking_labels_list:
        n_frames, n_instances, n_swimmer_instances, n_boat_instances = process_tracking_csv(tracking_label_path)

        video_key = Path(tracking_label_path).stem

        if video_key in TRAIN_VIDEO_KEY:
            row_idx = df.index[df["Split"] == "Train"][0]
        elif video_key in VAL_VIDEO_KEY:
            row_idx = df.index[df["Split"] == "Validation"][0]
        elif video_key in TEST_VIDEO_KEY:
            row_idx = df.index[df["Split"] == "Test"][0]

        # 7.2.1 Increment Split Row
        df.loc[row_idx, df.columns[1:]] += [1, n_frames, n_instances, n_swimmer_instances, n_boat_instances]

    # 7.2.2 Fill Total
    sums = df[df["Split"] != "Total"].iloc[:, 1:].sum().values

    df.loc[df["Split"] == "Total", df.columns[1:]] = [sums]

    # 7.2.3 Format with Percentages
    def format_percent_row(value, total):
        return f"{value:>7,d} ({(100 * value / total):6.2f}%)" if total > 0 else "      0 (  0.00%)"

    # Format swimmer
    df["Num. Swimmer"] = df.apply(
        lambda row: format_percent_row(row["Num. Swimmer"], row["Num. Instances"]),
        axis=1
    )

    # Format boat
    df["Num. Boat"] = df.apply(
        lambda row: format_percent_row(row["Num. Boat"], row["Num. Instances"]),
        axis=1
    )

    def format_percentage_column(column_name):
        df[column_name] = df[column_name].astype("object")

        total_instances = df.loc[df["Split"] != "Total", column_name].sum()

        df.loc[df["Split"] != "Total", column_name] = df.loc[df["Split"] != "Total", column_name].apply(
            lambda x: f"{int(x):>6,d} ({round(100 * x / total_instances):>6.2f}%)")

        df.loc[df["Split"] == "Total", column_name] = f"{int(total_instances):>6,d} (100.00%)"

    format_percentage_column("Num. Instances")
    format_percentage_column("Num. Frames")



    # 7.4 Print Dataset Analysis
    print(df)