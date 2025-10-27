import cv2
import torch
import pprint

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

def match_preds_targets_with_placeholders(preds, targets):
    """
    Match preds and targets by image_id, filling missing preds with empty placeholders.

    Returns aligned_preds and aligned_targets lists of same length.
    """
    # Helper to extract image_id
    def get_img_id(item):
        img_id = item.get('image_id', None)
        if img_id is None:
            raise ValueError("Missing 'image_id' in item.")
        if isinstance(img_id, torch.Tensor):
            return int(img_id[0].item()) if img_id.ndim > 0 else int(img_id.item())
        return int(img_id)

    # Build lookup dictionaries
    pred_dict = {get_img_id(p): p for p in preds}
    target_dict = {get_img_id(t): t for t in targets}

    # Get all image IDs from targets
    image_ids = sorted(target_dict.keys())

    aligned_preds = []
    aligned_targets = []

    for img_id in image_ids:
        target = target_dict[img_id]
        aligned_targets.append({
            "boxes": target["boxes"],
            "labels": target["labels"],
        })

        if img_id in pred_dict:
            # Prediction for this image
            pred = pred_dict[img_id]
            aligned_preds.append({
                "boxes": pred["boxes"],
                "scores": pred["scores"],
                "labels": pred["labels"],
            })
        else:
            # No prediction for this image -> placeholder empty pred
            aligned_preds.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long)
            })
    return aligned_preds, aligned_targets

# Load models
classifier = YOLO('models/yolo11n_MergedDataset_CLS.pt')
detector_coastline = YOLO('models/yolo12s_CoastlineDrone.pt')
detector_openwater = YOLO('models/yolo12s_SeaDroneSee.pt')

# Paths
gt_folder     = Path('datasets/CoastlineDrone-OD/labels/val') 
input_folder  = Path('datasets/CoastlineDrone-OD/images/val')
output_folder = Path('evaluation/MergedDataset-CLS-OD/val')
output_folder.mkdir(parents=True, exist_ok=True)

# Torchmetrics Mean Average Precision
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

# Get all image paths
image_paths = list(input_folder.glob('*.jpg'))

# Main pipeline
preds = []
targets = []

"""
For each image:
    1. Classify between coastline or open-water
    2. Obtain the predictions from the correct model and fill the preds list {'boxes':, 'scores':, 'labels'}
    3. Load and format the ground-truth of the corresponding image and fill the targets list {'boxes':, 'labels'}
Finally, evaluate the model using torchmetrics.
"""
for img_path in tqdm(image_paths):
    img_name = img_path.name
    img_stem = img_path.stem
    gt_path = gt_folder / f"{img_stem}.txt"

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    # --- Step 1: Classification --- #
    class_result = classifier.predict(img, imgsz=img_w, verbose=False)
    predicted_class = int(class_result[0].probs.top1)

    # --- Step 2: Detection --- #
    detection_model = detector_coastline if predicted_class == 0 else detector_openwater
    detection_result = detection_model.predict(img, save=False, verbose=False)

    for r in detection_result:
        # r.save(filename=str(output_folder / img_name))
        if r.boxes:
            boxes = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            labels = r.boxes.cls.cpu()
            preds.append({
                'boxes': boxes.to(dtype=torch.float32),
                'scores': scores.to(dtype=torch.float32),
                'labels': labels.to(dtype=torch.int64),
                'image_id': int(img_stem[6:])
            })

    # --- Step 3: Load and format ground-truth --- #
    tgt = [[], [], 0]   # boxes, labels, image_id
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            img_stem = img_path.stem
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
                    'image_id': int(img_stem[6:])
                })

aligned_preds, aligned_targets = match_preds_targets_with_placeholders(preds, targets)

print(f"Predictions for {len(aligned_preds)} images and Targets for {len(aligned_targets)} images.")

# Update metrics
metric.update(aligned_preds, aligned_targets)    
    
# Compute final metrics
final_metrics = metric.compute()

# Print results
print("\n=== Detection Metrics ===")
pprint.pprint(final_metrics)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # --- Prepare formats for torchmetrics ---
    # if preds:
    #     pred_boxes = torch.tensor(np.array([pred['box'] for pred in preds]), dtype=torch.float32)
    #     pred_scores = torch.tensor(np.array([pred['score'] for pred in preds]), dtype=torch.float32)
    #     pred_labels = torch.tensor(np.array([pred['class_id'] for pred in preds]), dtype=torch.int64)
    # else:
    #     pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
    #     pred_scores = torch.zeros((0,), dtype=torch.float32)
    #     pred_labels = torch.zeros((0,), dtype=torch.int64)

    # if gt_boxes_list:
    #     gt_boxes = torch.tensor(np.array([gt[:4] for gt in gt_boxes_list]), dtype=torch.float32)
    #     gt_labels = torch.tensor(np.array([gt[4] for gt in gt_boxes_list]), dtype=torch.int64)
    # else:
    #     gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
    #     gt_labels = torch.zeros((0,), dtype=torch.int64)

    # preds_dict = [{'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels}]
    # targets_dict = [{'boxes': gt_boxes, 'labels': gt_labels}]

    # preds (List): A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict
    # target (List): A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). Parameters that should be provided per dict:

    # Update metrics
    # metric.update(preds_dict, targets_dict)

# 1. Compute final metrics
# final_metrics = metric.compute()

# 3. Print results
# print("\n=== Detection Metrics ===")
# pprint.pprint(final_metrics)
# print(f"mAP@50: {final_metrics['map_50'].item():.4f}")
# print(f"mAP@50-95: {final_metrics['map'].item():.4f}")
# print(f"Recall (mar_100): {final_metrics['mar_100'].item():.4f}")