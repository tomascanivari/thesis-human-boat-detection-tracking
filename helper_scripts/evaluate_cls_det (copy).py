import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc


def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union

    return iou

def compute_ap(recall, precision):
    """Compute AP using the area under the precision-recall curve."""
    precision = np.insert(precision, 0, 1)
    recall = np.insert(recall, 0, 0)
    return auc(recall, precision)

def evaluate_map(ground_truths, predictions, iou_threshold=0.5):
    """
    Evaluate mAP.
    ground_truths: dict of {image_id: list of {'class_id': int, 'bbox': [x_min, y_min, x_max, y_max]}}
    predictions: dict of {image_id: list of {'class_id': int, 'confidence': float, 'bbox': [...]}}
    """
    aps = []
    
    for c in range(1, 3):
        true_positives = []
        scores = []
        total_gts = 0

        # Collect all predictions and ground truths for this class
        gt_records = defaultdict(list)
        pred_records = []
        
        for image_id in ground_truths.keys():
            gts = [gt for gt in ground_truths[image_id] if gt['class_id'] == c]
            preds = [pred for pred in predictions[image_id] if pred['class_id'] == c]
            
            total_gts += len(gts)
            gt_records[image_id] = {'boxes': [gt['bbox'] for gt in gts], 'detected': [False] * len(gts)}
            pred_records.extend([{'image_id': image_id, 'confidence': pred['confidence'], 'bbox': pred['bbox']} for pred in preds])
        
        # Sort predictions by confidence
        pred_records.sort(key=lambda x: x['confidence'], reverse=True)
        
        for pred in pred_records:
            image_id = pred['image_id']
            pred_box = pred['bbox']
            max_iou = 0
            max_idx = -1
            
            for idx, gt_box in enumerate(gt_records[image_id]['boxes']):
                iou = intersection_over_union(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            
            if max_iou >= iou_threshold:
                if not gt_records[image_id]['detected'][max_idx]:
                    true_positives.append(1)
                    gt_records[image_id]['detected'][max_idx] = True
                else:
                    true_positives.append(0)  # Duplicate detection
            else:
                true_positives.append(0)
            
            scores.append(pred['confidence'])
        
        if total_gts == 0:
            aps.append(0)
            continue
        
        # Compute precision-recall curve
        true_positives = np.array(true_positives)
        scores = np.array(scores)
        
        sorted_indices = np.argsort(-scores)
        true_positives = true_positives[sorted_indices]
        
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(1 - true_positives)
        
        recalls = cum_tp / total_gts
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        print(f"Class {c} AP: {ap:.4f}")
    
    mean_ap = np.mean(aps)
    print(f"\nMean AP (mAP): {mean_ap:.4f}")
    return aps, mean_ap

def cxcywh2xywh(bbox, img_sz):
    
    # Unpack bbox and img_sz
    img_width, img_height = img_sz
    cxn, cyn, wn, hn = bbox
    
    # Convert from cx, cy, w, h (normalized) to x, y, w, h (not normalized)
    w = wn * img_width
    h = hn * img_height
    x = (cxn - wn / 2) * img_width
    y = (cyn - hn / 2) * img_height

    return [x, y, w, h]

def read_txt_annotations(directory, img_sz):
    annotations = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            img_name = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as f:
                lines = f.readlines()

            objects = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = cxcywh2xywh(list(map(float, parts[1:])), img_sz)
                    objects.append({'class_id': class_id, 'bbox': bbox})
                elif len(parts) == 6:
                    class_id = int(parts[0])
                    conf = float(parts[5])
                    bbox = cxcywh2xywh(list(map(float, parts[1:5])), img_sz)
                    objects.append({'class_id': class_id, 'confidence': conf, 'bbox': bbox})
            annotations[img_name] = objects

    return annotations

img_sz = (640, 485)

gt_path   = "datasets/CoastlineDrone-OD/labels/val"
pred_path = "datasets/CoastlineDrone-OD/pred_labels/val"

gt_ann   = read_txt_annotations(directory=gt_path, img_sz=img_sz)
pred_ann = read_txt_annotations(directory=pred_path, img_sz=img_sz)

if __name__ == "__main__":
    evaluate_map(gt_ann, pred_ann)