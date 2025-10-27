import os
import csv
import shutil
import argparse

import numpy as np
import pandas as pd
import pprint as pp

from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)  # hardlink
    except OSError as e:
        print(f"Linking failed for {src} -> {dst}: {e}")

def train_classifier(data, name):
    
    model = YOLO("yolo11s-cls.pt")
    
    kwargs = {
        "data"       : data,
        "epochs"     : 5,
        "device"     : "cuda",
        "imgsz"      : 640,
        "save"       : True,
        "workers"    : 12,
        "batch"      : 12,
        "name"       : name,
        "exist_ok"   : True,
        "optimizer"  : 'SGD'
    }

    model.train(**kwargs)

def evaluate_classifier(model_path, dataset, fold):

    class_names = ["Coastline", "OpenWater"]

    model = YOLO(model_path)

    y_true = []
    y_pred = []

    # Class names
    class_names = ["Coastline", "OpenWater"]
    num_classes = len(class_names)

    # Run evaluation
    for split in ["train", "val", "test"]:
        test_results = model.val(split=split, save=False, verbose=False)
        shutil.rmtree(f"runs/classify/val")

        # Confusion matrix (rows=predicted, cols=true)
        cm_yolo = test_results.confusion_matrix.matrix

        y_true = []
        y_pred = []

        for true_idx in range(num_classes):                 # columns = true labels
            for pred_idx in range(num_classes):             # rows = predicted labels
                count = int(cm_yolo[pred_idx, true_idx])    # number of images predicted=pred_idx, true=true_idx
                y_true.extend([true_idx] * count)
                y_pred.extend([pred_idx] * count)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        print("Confusion Matrix:")
        
        # Convert to DataFrame
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # Rename the corner cell
        cm_df.index.name = "True/Predicted"
        cm_df.columns.name = None

        print(cm_df)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=6, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=6))

        report_df = pd.DataFrame(report).transpose()

        # Save report
        report_df.to_json(f"evaluation/classification/{split}/{dataset}/fold{fold}_metrics.json", index=False)

        # Save confusion matrix
        cm_df.to_csv(f"evaluation/classification/{split}/{dataset}/fold{fold}_cm.csv")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classification + detection on a dataset.")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--mode', choices=['train', 'evaluate'], help="Mode to run: 'train' or 'evaluate'")
    parser.add_argument('--fold', choices=['0', '1', '2', '3', '4', '5'], help="Fold to run: '0', '1', '2', '3', '4', '5'")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():
    
    # === 1. Parse the arguments === #
    args = parse_arguments()
    dataset = args.dataset
    fold = args.fold
    mode = args.mode
    
    # === 2. Verify dataset === #
    dataset_dir = Path(f"datasets/{dataset}Dataset")
    folds_dir = dataset_dir / "folds"
    fold_dir = folds_dir / f"fold{fold}"

    # 2.1 Get correspondent fold files
    fold_files = {'train': folds_dir / f"train_fold{fold}.txt",
                  'val': folds_dir / f"val_fold{fold}.txt",
                  'test': folds_dir / f"test_fold{fold}.txt"} 

    # 2.2 Get correspondent image paths per split
    imgs = {}
    for split, file in fold_files.items():
        with open(file, 'r') as f:
            imgs[split] = [Path(line.strip()) for line in f if line.strip()]

    # 2.3 Create hardlinks between files in 'images' and in the folder split 
    for split, img_paths in imgs.items():
        split_dir = fold_dir / split
        coastline_dir = split_dir / "Coastline"
        openwater_dir = split_dir / "OpenWater"

        for img_path in img_paths:
            img_name = img_path.name
            src = img_path
            dst = coastline_dir / img_name if "Coastline" in str(img_path) else openwater_dir / img_name

            if not src.exists() or dst.exists():
                continue
    
            link(src, dst)

    if mode == 'train':
        train_classifier(data=fold_dir, name=f"{dataset}Fold{fold}")
    elif mode == 'evaluate':
        model_path = f"models/{dataset}Models/fold{fold}.pt"
        evaluate_classifier(model_path, dataset, fold)

if __name__ == '__main__':
    main()