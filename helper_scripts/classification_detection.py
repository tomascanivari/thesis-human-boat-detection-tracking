import os
import sys
import csv
import cv2
import torch
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

from utils import evaluation_utils as ev


def get_filenames_without_extension(directory, extension_filter=None):
    files = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            name, ext = os.path.splitext(f)
            if extension_filter is None or ext.lower() == extension_filter.lower():
                files.append(name)
    return set(files)

def compare_and_exit_on_mismatch(dir1, ext1, dir2, ext2):
    files1 = get_filenames_without_extension(dir1, ext1)
    files2 = get_filenames_without_extension(dir2, ext2)

    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    if only_in_dir1 or only_in_dir2:
        print("Mismatch detected!")
        if only_in_dir1:
            print(f"Files in {dir1} missing in {dir2}: {sorted(only_in_dir1)}")
            for file in only_in_dir1:
                open(f"{dir2}/{file}.txt", 'w').close()

        if only_in_dir2:
            print(f"Files in {dir2} missing in {dir1}: {sorted(only_in_dir2)}")
        # sys.exit(1)  # Exit with error code

    # print(f"All Images have a Ground-Truth annotation ({len(files1)} images).")

def xywhn2xyxy(bbox, img_w, img_h):
    """ 
    Convert normalized [cxn, cyn, wn, hn] to [x1, y1, x2, y2] pixel coordinates .
    
    x1: x-axis of the top-left corner
    y1: y-axis of the top-left corner
    x2: x-axis of the bottom-right corner
    y2: y-axis of the bottom-right corner
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classification + detection on a dataset.")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--split', type=str, required=True, help="Split of the dataset.")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment (if it already exists all data is overwritten).")

    # Optional arguments
    parser.add_argument('--model_opt', type=int, required=False, help="Model options: 0 - Both (w/ cls); 1 - Coastline (/wo cls); 2 - Open-water (/wo cls)", default=0)

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():

    #####################################
    # --- Step 1: Process Arguments --- #
    #####################################
    args = parse_arguments()

    split = args.split          # Split of the dataset ('train' or 'val')
    dataset = args.dataset      # Dataset name (assumes it has the correct structure and is in correct directory '/datasets/')
    exp_name = args.exp_name    # Experiment directory name (/runs/classify_detection/{exp_name})
    model_opt = args.model_opt  # Models behaviour (0 - Both w/ cls; 1 - Coastline /wo cls; 2 - Open-water /wo cls)

    ################################################
    # --- Step 2: Verify Directories Integrity --- #
    ################################################
    exp_folder    = Path(f'runs/classify_detection/{exp_name}')
    gts_folder    = Path(f'datasets/{dataset}/labels/{split}') 
    images_folder = Path(f'datasets/{dataset}/images/{split}')
    
    # Check files in image_folder and gts_folder for integrity
    compare_and_exit_on_mismatch(images_folder, '.jpg', gts_folder, '.txt')
    
    # Load all image paths
    image_paths = list(images_folder.glob('*.jpg'))

    # Create and/or clear the experiment folder
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    exp_folder.mkdir(parents=True, exist_ok=True)
    
    # Fill the experiment folder with the necessary sub-folders
    save_images_folder = exp_folder / "images"      # Folder that stores the annotated images 
    pred_labels_folder = exp_folder / "pred_labels" # Folder that stores the predicted labels

    save_images_folder.mkdir()
    pred_labels_folder.mkdir()

    metrics_path = str(exp_folder / "detection_metrics.csv") 

    ###############################################
    # --- Step 3: Load Models and Run Options --- #
    ###############################################
    do_cls = True   
    detector_no_cls : YOLO

    classifier         = YOLO('models/yolo11n_MergedDataset_CLS.pt')
    detector_merged    = YOLO('models/yolo12s_Merged.pt')
    detector_coastline = YOLO('models/yolo12s_CoastlineDrone.pt')
    detector_openwater = YOLO('models/yolo12m_SeaDroneSee.pt')
    # detector_coastline = YOLO('runs/detect/CoastlineDatasetFold0/weights/last.pt')
    
    if model_opt == 1:
        # No classification and only Coastline model
        do_cls = False
        detector_no_cls = detector_coastline
    elif model_opt == 2:
        # No classification and only Open-water model
        do_cls = False
        detector_no_cls = detector_openwater
    elif model_opt == 3:
        # No classification and only Open-water model
        do_cls = False
        detector_no_cls = detector_merged

    ######################################################
    # --- Step 4: Show and Store Experiment Settings --- #
    ######################################################
    
    file = open(str(exp_folder / "settings.txt"), 'w')

    mopt2str = {0:"classification active and both models", 
                1:"classification not active and only coastline model", 
                2:"classification not active and only open-water model",
                3:"classification not active and only merged model"}

    for f in [sys.stdout, file]:
        if f == sys.stdout:
            start = "\033[1m"
            end = "\033[0m"
        elif f == file:
            start, end = "", ""
        print(f"{start}EXPERIMENT {exp_name} SETTINGS:{end}", file=f)
        print(f"             split = {start}{split}{end}", file=f)
        print(f"           dataset = {start}{dataset}{end}", file=f)
        print(f"         model_opt = {start}{mopt2str[model_opt]}{end}", file=f)
        print(f"        exp_folder = {start}{exp_folder}{end}", file=f)
        print(f"        gts_folder = {start}{gts_folder}{end}", file=f)
        print(f"        img_folder = {start}{images_folder}{end}", file=f)
        print(f"save_images_folder = {start}{save_images_folder}{end}", file=f)
        print(f"pred_labels_folder = {start}{pred_labels_folder}{end}", file=f)
        
    
    #################################
    # --- Step 5: Main Pipeline --- #
    #################################
    count = [0, 0]  # Number of images per class
    preds   = []    # preds   = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...] 
    targets = []    # targets = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N)}, {...}, ...] 
    
    print(f"\n\n\033[1mRUNNING ON {dataset.upper()} {split.upper()}\033[0m")

    # Process each image one by one
    for img_path in tqdm(image_paths):
        img_name = img_path.stem
        if img_path.name.startswith("frame_"):
            img_name = int(img_path.stem[6:])

        # Read image and store width and height
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        # Classification
        if do_cls:
            # Predict class (coastline or openwater)
            class_result = classifier.predict(img, imgsz=img_w, verbose=False)
            predicted_class = int(class_result[0].probs.top1)

            # Detection model
            detection_model = detector_coastline if predicted_class == 0 else detector_openwater
            count[predicted_class] += 1 
        else:
            # Detection model
            detection_model = detector_no_cls

        # Detection
        detection_result = detection_model.predict(img, save=False, verbose=False, classes=[1, 2])

        # Process the detections
        for r in detection_result:
            # Save the image
            r.save(filename=str(save_images_folder / img_path.name))
            # Store predicted labels in format {'image_id': (1), 'label': (N), 'boxes': (N, 4), 'scores': (N)}
            # with boxes format xyxy
            if r.boxes:
                boxes   = r.boxes.xyxy.cpu()
                scores  = r.boxes.conf.cpu()
                labels  = r.boxes.cls.cpu()
                boxes_n = r.boxes.xywhn.cpu().numpy()
                preds.append({
                    'boxes': boxes.to(dtype=torch.float32),
                    'scores': scores.to(dtype=torch.float32),
                    'labels': labels.to(dtype=torch.int64),
                    'image_id': int(img_name)
                })

                # Save predicted labels in YOLO format [label cxn cyn wn hn score]
                with open(file=str(pred_labels_folder / str(img_path.stem + ".txt")), mode="w") as f:
                    for label, box, score in zip(labels, boxes_n, scores):
                        f.write(f"{int(label)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {score:.4f}")

        # Load and format ground-truth --- #
        gt_path = gts_folder / f"{img_path.stem}.txt"
        tgt = [[], [], 0]   # boxes (xyxy), labels, image_id
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    elems = line.strip().split()
                    label = int(elems[0])
                    cx, cy, w, h = map(float, elems[1:])
                    box = xywhn2xyxy([cx, cy, w, h], img_w, img_h)
                    tgt[0].append(box)
                    tgt[1].append(label)     
                targets.append({
                        'boxes': torch.tensor(tgt[0], dtype=torch.float32),
                        'labels': torch.tensor(tgt[1], dtype=torch.int64),
                        'image_id': int(img_name)
                    })

    # Print class statistics
    if do_cls:
        print("\n\033[1mCLASSIFICATION\033[0m")
        print(f"Number of Coastline  images: {count[0]}")
        print(f"Number of Open-water images: {count[1]}")

    ##############################
    # --- Step 6: Evaluation --- #
    ##############################
    print("\n\033[1mDETECTION METRICS\033[0m")
    det_summary = ev.evaluate_detection(preds, targets, metrics_path)
    print(det_summary)

if __name__ == "__main__":
    main()