import cv2
import torch
import pprint
import numpy as np

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def xywhn2xyxy(bbox, img_w, img_h):
    """ Convert normalized [cx, cy, w, h] to [x1, y1, x2, y2] pixel coordinates """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]

# Load models
classifier = YOLO('models/yolo11n_MergedDataset_CLS.pt')
detector_coastline = YOLO('models/yolo12s_CoastlineDrone.pt')
detector_openwater = YOLO('models/yolo12s_SeaDroneSee.pt')

# Paths
gt_folder     = Path('datasets/test/labels/val') 
input_folder  = Path('datasets/test/images/val')
output_folder = Path('evaluation/MergedDataset-CLS-OD/val')
output_folder.mkdir(parents=True, exist_ok=True)

# Torchmetrics Mean Average Precision
metric = MeanAveragePrecision(iou_type="bbox", extended_summary=True)

# Get all image paths
image_paths = list(input_folder.glob('*.jpg'))

# Main pipeline
for img_path in tqdm(image_paths):
    img_name = img_path.name
    img_stem = img_path.stem
    gt_path = gt_folder / f"{img_stem}.txt"

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    # Step 1: Classification
    class_result = classifier.predict(img, imgsz=640, verbose=False)
    predicted_class = int(class_result[0].probs.top1)

    # Step 2: Detection
    detection_model = detector_coastline if predicted_class == 0 else detector_openwater
    detection_result = detection_model.predict(img, save=False, verbose=False)

    # Save detection output image
    for r in detection_result:
        # r.save(filename=str(output_folder / img_name))

        preds = []
        if r.boxes:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsss = r.boxes.cls.cpu().numpy()
            for box, conf, clss in zip(boxes, confs, clsss):
                preds.append({
                    'box': box,
                    'score': conf,
                    'class_id': int(clss)
                })

    # --- Load and format ground-truth boxes ---
    gt_boxes_list = []
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                elems = line.strip().split()
                cls_id = int(elems[0])
                cx, cy, w, h = map(float, elems[1:])
                box = xywhn2xyxy([cx, cy, w, h], img_w, img_h)
                gt_boxes_list.append(box + [cls_id])

    # --- Prepare formats for torchmetrics ---
    if preds:
        pred_boxes = torch.tensor(np.array([pred['box'] for pred in preds]), dtype=torch.float32)
        pred_scores = torch.tensor(np.array([pred['score'] for pred in preds]), dtype=torch.float32)
        pred_labels = torch.tensor(np.array([pred['class_id'] for pred in preds]), dtype=torch.int64)
    else:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
        pred_labels = torch.zeros((0,), dtype=torch.int64)

    if gt_boxes_list:
        gt_boxes = torch.tensor(np.array([gt[:4] for gt in gt_boxes_list]), dtype=torch.float32)
        gt_labels = torch.tensor(np.array([gt[4] for gt in gt_boxes_list]), dtype=torch.int64)
    else:
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_labels = torch.zeros((0,), dtype=torch.int64)

    preds_dict = [{'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels}]
    targets_dict = [{'boxes': gt_boxes, 'labels': gt_labels}]

    # Update metrics
    metric.update(preds_dict, targets_dict)

# 1. Compute final metrics
final_metrics = metric.compute()

# 3. Print results
print("\n=== Detection Metrics ===")
pprint.pprint(final_metrics)
# print(f"mAP@50: {final_metrics['map_50'].item():.4f}")
# print(f"mAP@50-95: {final_metrics['map'].item():.4f}")
# print(f"Recall (mar_100): {final_metrics['mar_100'].item():.4f}")