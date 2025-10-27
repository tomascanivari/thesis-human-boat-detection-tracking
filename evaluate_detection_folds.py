import os
import re
import cv2
import torch
import shutil
import argparse

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

from utils import evaluation_utils as ev
from utils import dataset_analysis as da

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

def load_folds_paths(folds_dir):
    
    # Folds txt files
    txt_files = sorted(glob(os.path.join(folds_dir, '*.txt')))

    # Paths
    image_paths = defaultdict(dict)
    label_paths = defaultdict(dict)

    pattern = re.compile(r'(train|val|test)_fold(\d+)')

    # Obtain paths from folds files
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        match = pattern.match(filename)
        if match:
            split = match.group(1)
            fold = int(match.group(2))
            with open(txt_file, 'r') as f:
                imgs = [line.strip() for line in f if line.strip()]
                lbls = [p.replace('/images', '/labels') for p in imgs]
                lbls = [p.replace('jpg', 'txt') for p in lbls]
                image_paths[split][fold] = imgs
                label_paths[split][fold] = lbls

    return image_paths, label_paths

def load_detection_models(models_dirs):
    
    models = {}

    for path in models_dirs:

        model_name = os.path.basename(path)

        models[model_name] = {}

        model_files = sorted(glob(os.path.join(path, 'fold*.pt')))

        pattern = re.compile(r'fold(\d+)\.pt')

        for model_file in model_files:

            match = pattern.search(os.path.basename(model_file))
            if match:
                fold = int(match.group(1))
                models[model_name][fold] = YOLO(model_file)
        
        # CoastlineV2 only has 1 model (fold0)
        if model_name == "CoastlineV2Models":
            for fold in range(1, 6):
                models[model_name][fold] = models[model_name][0]

    return models

def load_classification_models(v2_flag):

    classification_models = {}

    models_str = "ClassificationV2Models" if v2_flag else "ClassificationModels"

    model_files = sorted(glob(os.path.join(Path(f"models/{models_str}"), 'fold*.pt')))

    pattern = re.compile(r'fold(\d+)\.pt')

    for model_file in model_files:
        match = pattern.search(os.path.basename(model_file))
        if match:
            fold = int(match.group(1))
            classification_models[fold] = YOLO(model_file)
    
    return classification_models

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classification + detection on a dataset.")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--models', type=str, required=True, help="Models to use.")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment (if it already exists all data is overwritten).")

    # Optional
    parser.add_argument('--split', type=str, required=False, help="Run only on this split.", default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():

    #####################################
    # === Step 1: Process Arguments === #
    #####################################
    args = parse_arguments()
    dataset = args.dataset      # Dataset name (assumes it has the correct structure and is in correct directory '/datasets/')
    models_dataset = args.models
    exp_name = args.exp_name    # Experiment directory name (/runs/classify_detection/{exp_name})
    split = args.split

    do_classification = False

    if split is None:
        # splits = ['train', 'val', 'test']
        splits = ['train', 'val']
    else:
        splits = [split]

    ################################################
    # === Step 2: Verify Directories Integrity === #
    ################################################
    exp_folder    = Path(f'runs/detection/{exp_name}')  

    # Create and/or clear the experiment folder
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    exp_folder.mkdir(parents=True, exist_ok=True)
    
    #########################################
    # === Step 3: Load Models and Paths === #
    #########################################
    if models_dataset == "Classification" or models_dataset == "ClassificationV2" or models_dataset == "ClassificationV2Big":
        do_classification = True
        coastline_models = "Coastline"
        coastline_models = coastline_models + "V2" if "V2" in models_dataset else coastline_models
        coastline_models = coastline_models + "Big" if "Big" in models_dataset else coastline_models
        detection_models = load_detection_models(models_dirs=[Path(f"models/{coastline_models}Models"), Path("models/OpenWaterModels")])
        
        v2_flag = True if "V2" in coastline_models else False
        classification_models = load_classification_models(v2_flag)

    elif models_dataset in ['Coastline', 'CoastlineV2', 'CoastlineV2Big', 'OpenWater', 'Merged']:
        do_classification = False
        detection_models = load_detection_models(models_dirs=[Path(f"models/{models_dataset}Models")])

    image_paths, label_paths = da.dataset_analysis(dataset)

    ######################################################################################
    # === Step 4: Classification + Detection + Evaluation Pipeline By Fold and Split === #
    ######################################################################################
    dfs_metrics = []
    for fold in [0, 1, 2, 3, 4, 5]:
        for split in splits:
            print(f"\n\n\033[1mRUNNING {models_dataset.upper()} ON {dataset.upper()} FOLD {fold} {split.upper()}\033[0m")

            # Fill the experiment folder with the necessary sub-folders
            save_images_folder = exp_folder / f"fold_{fold}" / f"{split}" / "images"      # Folder that stores the annotated images 
            pred_labels_folder = exp_folder / f"fold_{fold}" / f"{split}" / "pred_labels" # Folder that stores the predicted labels

            save_images_folder.mkdir(parents=True)
            pred_labels_folder.mkdir(parents=True)

            metrics_path = str(exp_folder / f"fold_{fold}" / f"{split}" / "detection_metrics.csv") 

            count = [0, 0]  # Number of images per class
            preds   = []    # preds   = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...] 
            targets = []    # targets = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N)}, {...}, ...] 

            # Process Each Image & Label
            for img_path, label_path in tqdm(zip(image_paths[split][fold], label_paths[split][fold]), total=len(image_paths[split][fold])):
                
                img_path = Path(img_path)

                img_name = img_path.stem
                if img_path.name.startswith("frame_"):
                    img_name = int(img_path.stem[6:])                       

                # Read image and store width and height
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]

                # Classification
                if do_classification:
                    # Predict class (coastline or openwater)
                    class_result = classification_models[fold].predict(img, imgsz=img_w, verbose=False, stream=True)
                    class_result = next(class_result)  # get the first (and only) result
                    predicted_class = int(class_result[0].probs.top1)
                    # predicted_class = 0 if "Coastline" in str(img_path) else 1
                    # Detection model
                    detection_model = detection_models[f"{coastline_models}Models"][fold] if predicted_class == 0 else detection_models["OpenWaterModels"][fold]
                    count[predicted_class] += 1 
                else:
                    # Detection model
                    detection_model = detection_models[f"{models_dataset}Models"][fold]

                # Detection
                detection_result = detection_model.predict(img, save=False, verbose=False, classes=[1, 2], stream=True)  # 1 -> swimmer, 2 -> boat

                # Process the detections
                for r in detection_result:
                    # Save the image
                    # r.save(filename=str(save_images_folder / img_path.name))
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
                                f.write(f"{int(label)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {score:.4f}\n")

                # Load and format ground-truth
                tgt = [[], [], 0]   # boxes (xyxy), labels, image_id
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        elems = line.strip().split()
                        if len(elems) != 5:
                            # Corrupted label
                            continue
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

            
            # === Step 4.1: Evaluation Per Fold === #
            # Print Class Statistics
            if do_classification:
                print("\n\033[1mCLASSIFICATION\033[0m")
                print(f"Number of Coastline  images: {count[0]}")
                print(f"Number of Open-water images: {count[1]}")            

            # Print Detection Metrics
            print("\n\033[1mDETECTION METRICS\033[0m")
            df = ev.evaluate_detection(preds, targets, metrics_path)  
                
            # Reset index to make 'class' a column
            df = df.reset_index().rename(columns={'index': 'class'})

            # Define required classes
            required_classes = ['swimmer', 'boat']

            # Ensure all required classes are present
            existing_classes = set(df['class'])
            missing_classes = set(required_classes) - existing_classes

            # Append missing rows with NaNs
            columns = df.columns.tolist()
            for class_name in missing_classes:
                row = {col: np.nan for col in columns}
                row['class'] = class_name
                row['num_preds'] = 0
                row['num_gts'] = 0
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            # Sort by class name
            df = df.sort_values('class')

            # Append Dataset, Model, Split and Fold
            df.insert(0, 'fold', fold)
            df.insert(0, 'split', split)
            df.insert(0, 'model', models_dataset)
            df.insert(0, 'dataset', dataset)

            # Save to csv
            df.to_csv(metrics_path, index=False)  # Save with the index  

            print(df)
            dfs_metrics.append(df.copy())
    
    #########################################################
    # === Step 5: Create CSV With All Detection Metrics === #
    #########################################################
    df = pd.concat(dfs_metrics)
    
    # Define custom order for 'split'
    split_order = ['train', 'val', 'test']
    df['split'] = pd.Categorical(df['split'], categories=split_order, ordered=True)

    # Sort by 'split' then by 'fold' (assumed to be int)
    df = df.sort_values(['split', 'fold'])
    
    df.to_csv(exp_folder / "all_detection_metrics.csv", index=False)
    
    print("")
    print(df)

#############
# Main Call #
#############
if __name__ == "__main__":
    main()