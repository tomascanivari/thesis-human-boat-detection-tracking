
import os
import re
import torch
import numpy as np
import pandas as pd
import motmetrics as mm
from sklearn.metrics import auc
from collections import defaultdict
import matplotlib.pyplot as plt

def match_preds_targets_with_placeholders(preds, targets):
    """
    Match preds and targets by image_id, filling missing preds with empty placeholders.

    - preds   = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...] 
    - targets = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N)}, {...}, ...]

    Returns aligned_preds and aligned_targets lists of same length.

    - aligned_preds    = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...] 
    - aligned_targets = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'detected': (N)}, {...}, ...]
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

    n_gts = 0
    n_preds = 0

    for img_id in image_ids:
        target = target_dict[img_id]
        aligned_targets.append({
            "boxes": target["boxes"],
            "labels": target["labels"],
            "image_id": target["image_id"],
            'detected': [False] * len(target["labels"])
        })

        n_gts += len(target["labels"]) 

        if img_id in pred_dict:
            # Prediction for this image
            pred = pred_dict[img_id]
            aligned_preds.append({
                "boxes": pred["boxes"],
                "scores": pred["scores"],
                "labels": pred["labels"],
                "image_id": pred["image_id"]
            })

        else:
            # No prediction for this image -> placeholder empty pred
            aligned_preds.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long),
                "image_id": img_id
            })

    return aligned_preds, aligned_targets

def intersection_over_union(box1, box2):
    """
    Compute IoU between two boxes.
    
    - box1: (x1, y1, x2, y2)
    - box2: (x1, y1, x2, y2)

    Where:
    - x1: x-axis of the top-left corner
    - y1: y-axis of the top-left corner
    - x2: x-axis of the bottom-right corner
    - y2: y-axis of the bottom-right corner
    """

    # Intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # No overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Areas of the two boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU calculation
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def compute_ap(recall, precision):
    """Compute AP using the area under the precision-recall curve."""
    
    # Add (0,1) start point and (1,0) end point
    recalls = np.concatenate(([0.0], recall, [1.0]))
    precisions = np.concatenate(([1.0], precision, [0.0]))

    # Step 2: Make precision non-increasing (interpolation)
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # Step 3: Remove points where recall does not change (to avoid zero-width area)
    ap = auc(recalls, precisions)

    return ap

def plot_pr_curves(classes, labels_exist, precisions_list, recalls_list, f1_scores_list, save_path="pr_curve.png"):
    """
    Plot PR curves for multiple classes with max F1 points and save the figure.
    
    Args:
        classes (list of str): Class names.
        precisions_list (list of np.array): List of precision arrays (one per class).
        recalls_list (list of np.array): List of recall arrays (one per class).
        f1_scores_list (list of np.array): List of F1-score arrays (one per class).
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    
    for i, cls in enumerate(classes):
        if labels_exist[i]:
            precisions = precisions_list[i]
            recalls = recalls_list[i]
            f1_scores = f1_scores_list[i]

            # Plot the PR curve
            plt.plot(recalls, precisions, label=f'{cls} PR curve')
            
            # Find max F1 index and mark the point
            max_f1_idx = np.argmax(f1_scores)
            plt.scatter(recalls[max_f1_idx], precisions[max_f1_idx], 
                        color='red', marker='o', s=100, 
                        label=f'{cls} max F1 = {f1_scores[max_f1_idx]:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves with Max F1 Points')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path)

def tracking2detection(gt_file, pred_file):
    """
    Converts tracking results ['frame_id', 'category_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
    to detection results [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...], where boxes
    are [x1, y1, x2, y2] instead of [x, y, w, h], labels are the categories and scores are the confidences.
    """
    
    # Create the returned lists
    return_lists = []   # 0 : target, 1 : predictions
    
    # Load files data-frames 
    gt_df   = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)

    # Group by 'frame_id' and store each group as a dictionary
    for df in [gt_df, pred_df]:
        # Filter the unnecesary columns
        df = df.drop(columns=['track_id', 'xx', 'yy', 'zz'])
        
        # Convert from w, h to x2, y2
        df['w'] = df['x'] + df['w']
        df['h'] = df['y'] + df['h']

        # Concatenate the 'x', 'y', 'w', 'h' columns into a single 'coords' column
        df['boxes'] = df[['x', 'y', 'w', 'h']].apply(lambda row: torch.tensor(list(row)), axis=1)

        # Drop the original 'x', 'y', 'w', 'h' columns if you no longer need them
        df = df.drop(columns=['x', 'y', 'w', 'h'])

        df = df.rename(columns={'frame_id': 'image_id', 'category_id':'labels', 'conf':'scores'})

        # Create a dictionary for each image_id in the correct format and dimensions
        grouped_dicts = []
        for image_id, group in df.groupby('image_id'):
            item = {'image_id': image_id}
            for col in group.columns:
                if col == 'image_id':
                    continue
                
                if col == 'boxes':
                    item[col] = torch.stack(group[col].tolist())
                    continue
                
                item[col] = torch.tensor(group[col].tolist())
            
            grouped_dicts.append(item)
        
        # Append the dictionary with all images_id
        return_lists.append(grouped_dicts)

    # Return predictions, targets
    return return_lists[1], return_lists[0]

def evaluate_detection(preds, targets, metrics_path, iou_threshold=0.5):
    
    # Align the list of dictionaries (element i corresponds to the same image i)
    # predictions    = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'scores': (N)}, {...}, ...] 
    # ground_truths  = [{'image_id': (1), 'boxes': (N, 4), 'labels': (N), 'detected': (N)}, {...}, ...] 
    predictions, ground_truths = match_preds_targets_with_placeholders(preds, targets)

    # Compute metrics
    
    # Average Precision (for all IoU thresholds) per label
    aps = [[], []]

    # Precision and Recall that maximize the F1-score at IoU = 0.5 per label
    precision = [0, 0]
    recall = [0, 0]
    f1_scores = [0, 0]
    num_instances = [0, 0]

    # True Positives array (Nx10) where 1 marks a TP and 0 marks a FP
    tps = []
    
    # Scores of the detections (N)
    scores = []

    # Debug [(N,4), (N,4), (N), (N)]
    pred_labels = []

    # For each image with predictions (all since when there are no predictions a placeholder is in effect)
    for pred, gt in zip(predictions, ground_truths):

        # Sort prediction by score
        sorted_indices = torch.argsort(pred['scores'], descending=True)

        pred['boxes'] = pred['boxes'][sorted_indices]
        pred['labels'] = pred['labels'][sorted_indices]
        pred['scores'] = pred['scores'][sorted_indices]

        if torch.all(pred['labels'] == 0):
            continue

        # For each prediction find the best matching ground-truth 
        for pred_box, pred_score, pred_label in zip(pred['boxes'], pred['scores'], pred['labels']):
            
            scores.append(pred_score)
            pred_labels.append(pred_label)

            max_iou = 0
            max_idx = -1

            # Try to match prediction to ground-truths
            for idx, (gt_box, gt_label) in enumerate(zip(gt['boxes'], gt['labels'])):
                
                # Only try to match if the boxes are of the same class
                if pred_label == gt_label:

                    iou = intersection_over_union(gt_box, pred_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx

            if max_idx == -1:
                # No ground-truth of the same label
                tps.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                continue

            # Check if the best iou is above the treshold to determine if it is TP or FP
            if max_iou >= iou_threshold:
                if not gt['detected'][max_idx]:
                    # Matched -> Mark as TP for all IoU thresholds it is bigger than
                    pred_tps = []
                    for iou_trshld in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                        if max_iou >= iou_trshld:
                            pred_tps.append(1)
                        else:
                            pred_tps.append(0)
                    tps.append(torch.tensor(pred_tps))
                    gt['detected'][max_idx] = True
                else:
                    # Duplicate detection -> Mark as FP for all IoU thresholds
                    tps.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))   
            else:
                # Not matched -> Mark as FP for all IoU thresholds
                tps.append(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) 

    # Convert all to tensors with correct dimensions
    tps = torch.stack(tps)
    scores = torch.stack(scores)
    pred_labels = torch.stack(pred_labels)

    # Compute AP per label and IoU threshold
    labels_exist = [True, True]
    total_gts_per_label = [0, 0]
   
    # Count ground-truths for each label
    for gt in ground_truths:
        labels = gt['labels']
        total_gts_per_label[0] += (labels == 1).sum().item()
        total_gts_per_label[1] += (labels == 2).sum().item()

    # For PR Curve
    precisions_by_class = []
    recalls_by_class    = []
    f1_scores_by_class  = []
    classes_names       = ['swimmer', 'boat']

    for label in [1, 2]:
        # Label Mask
        mask_label = pred_labels == label
  
        num_instances[label-1] = mask_label.sum() 

        # Check if there is any prediction for this label
        if not mask_label.any(): 
            labels_exist[label-1] = False
            aps[label-1].extend([0] * 10)  
            continue
        
        for i in range(10):
            # Filter TP (for this IoU threshold), Scores and Total Ground-Truths
            tps_per_label = tps[:, i][mask_label]
            scores_per_label = scores[mask_label]

            # Compute precision-recall curve and AP@{IoU_threshold}
            sorted_indices = np.argsort(-scores_per_label)
            tps_per_label = tps_per_label[sorted_indices]
            
            cum_tp = np.cumsum(tps_per_label)
            cum_fp = np.cumsum(1 - tps_per_label)
            
            recalls = cum_tp / total_gts_per_label[label-1]
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

            # For IoU = 0.5 retrieve precision and recall that give the maximum F1-Score
            if i == 0:
                f1_scores_arr = 2 * (precisions * recalls) / (precisions + recalls + 1e-8) 
                idx = torch.argmax(f1_scores_arr)
                f1_scores[label-1] = f1_scores_arr[idx]
                precision[label-1] = precisions[idx]
                recall[label-1] = recalls[idx]

                # For PR Curve
                recalls_by_class.append(recalls.numpy())
                precisions_by_class.append(precisions)
                f1_scores_by_class.append(f1_scores_arr.numpy())

            ap = compute_ap(recalls, precisions)
            
            # Store AP@{IoU_threshold} for this label
            aps[label-1].append(ap)

    plot_pr_curves(
        classes_names,
        labels_exist,
        precisions_by_class,
        recalls_by_class,
        f1_scores_by_class,
        save_path=metrics_path.replace("detection_metrics.csv", "pr_curve.png")
    )

    # Convert aps [2x10] (one AP for each label and IoU treshold) to tensor
    aps = torch.tensor(aps)

    # Mask metrics by existing labels only
    aps = aps[labels_exist]
    precision = torch.tensor(precision)[labels_exist]
    recall = torch.tensor(recall)[labels_exist]
    f1_scores = torch.tensor(f1_scores)[labels_exist]
    num_instances = torch.tensor(num_instances)[labels_exist]
    total_gts_per_label = torch.tensor(total_gts_per_label)[labels_exist]

    # Mean Average Precision 50 per label 
    maps50 = aps[:, 0]

    # Mean Average Precision 50 95 per label
    maps5095 = aps.mean(dim=1)    

    # Define metrics list of tensors (num_existing_labels)
    metrics = [num_instances, total_gts_per_label, precision, recall, f1_scores, maps50, maps5095] 

    # Convert each tensor to a list, then zip to transpose rows into columns
    data = list(zip(*[t.tolist() for t in metrics]))

    # Create DataFrame: each tensor is a column
    df = pd.DataFrame(data, columns=['num_preds', 'num_gts', 'precision', 'recall', 'f1-score', 'mAP@50', 'mAP@50_95'])

    # Compute sum for first two columns, mean for the rest
    first_two_sums = df.iloc[:, :2].sum()
    other_means = df.iloc[:, 2:].mean()

    # Concatenate results into a single Series
    summary_row = pd.concat([first_two_sums, other_means])
    
    # Insert the row at the top and format for print
    summary = pd.concat([summary_row.to_frame().T, df], ignore_index=True)
    
    labels_names = ['swimmer', 'boat']
    add_to_index = []
    for label_exist, label_name in zip(labels_exist, labels_names):
        if label_exist:
            add_to_index.append(label_name)
     
    summary.index = ['all'] + add_to_index
    summary['num_preds'] = summary['num_preds'].astype('int')
    summary['num_gts'] = summary['num_gts'].astype('int')

    return summary

def evaluate_tracking(gt_path, pred_path, metrics_path, size=(640, 485)):
    """
    Evaluate tracking results using motmetrics.
    
    :param gt_file: Path to ground truth CSV file.
    :param pred_file: Path to predicted tracking CSV file.
    """
    
    cid2label = {0: "all", 1: 'swimmer', 2: 'boat'}   # Category_id to label name
    
    # Load files to lists of data-frames 
    gt_df   = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # Get sorted list of all category_ids present in either dataframe
    category_ids = sorted(set(gt_df['category_id'].unique()).union(pred_df['category_id'].unique()))

    gt_dfs = []
    pred_dfs = []

    for cat_id in category_ids:
        gt_dfs.append(gt_df[gt_df['category_id'] == cat_id].copy())
        pred_dfs.append(pred_df[pred_df['category_id'] == cat_id].copy())

    # Compute metrics for each class and then for all classes
    mh = mm.metrics.create()
    all_accs = []
    all_names = []

    total_raw = {
        'num_frames': 0,
        'num_objects': 0,
        'num_detections': 0,
        'num_misses': 0,
        'num_false_positives': 0,
        'num_switches': 0,
    }

    # For each class obtain an accumulator with the gts, preds and distances
    for i, cat_id in enumerate(category_ids):
        gt_df = gt_dfs[i]
        pred_df = pred_dfs[i]

        acc = mm.MOTAccumulator(auto_id=False)
        frames = sorted(gt_df['frame_id'].unique())
        
        for frame in frames:
            gt_frame = gt_df[gt_df['frame_id'] == frame]
            pred_frame = pred_df[pred_df['frame_id'] == frame]

            gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
            pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values
            gt_ids = gt_frame['track_id'].tolist()
            pred_ids = pred_frame['track_id'].tolist()

            dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

            acc.update(gt_ids, pred_ids, dist, frame)
        
        all_accs.append(acc)
        all_names.append(f'{cid2label[cat_id]}')

    # Compute full metrics for each class
    metrics = [
        'num_frames', 'num_objects', 'num_detections',
        'num_misses', 'num_false_positives', 'num_switches',
        'mota', 'motp', 'idf1',
    ]

    # List of summary dfs per class
    class_dfs = []

    # Compute metrics for each class
    for acc, name in zip(all_accs, all_names):
        class_dfs.append(mh.compute(acc, metrics=metrics, name=name))

    # Aggregate raw totals from class_df
    for cat_id, class_df in enumerate(class_dfs):
        class_df.insert(0, 'class', cid2label[cat_id+1])
        for metric in ['num_frames', 'num_objects', 'num_detections',
                   'num_misses', 'num_false_positives', 'num_switches']:
            total_raw[metric] += class_df[metric].sum()

    # total_raw['num_frames'] = class_dfs[0]['num_frames'].values[0] + class_dfs[1]['num_frames'].values[0]
    
    # Aggregate MOTP components manually
    sum_overlap_col = [0,] # All, Swimmer, Boat
    num_matches_col = [0,]
    for i, acc in enumerate(all_accs):
        events = acc.events.loc[acc.events.Type == 'MATCH'] # Filter matched bbox
        sum_overlap_col.append(events['D'].sum()) 
        num_matches_col.append(len(events))
         
    sum_overlap_col[0] = sum(sum_overlap_col[1:])
    num_matches_col[0] = sum(num_matches_col[1:])


    # Compute combined metrics
    FN   = total_raw['num_misses']
    FP   = total_raw['num_false_positives']
    IDSW = total_raw['num_switches']
    GT   = total_raw['num_objects']
    TP   = total_raw['num_detections']

    total_mota = 1 - (FN + FP + IDSW) / GT if GT > 0 else float('nan')
    total_motp = sum_overlap_col[0] / num_matches_col[0] if num_matches_col[0] > 0 else float('nan')
    total_idf1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else float('nan')

    # Create combined data frame
    summary_df = pd.DataFrame([{
        'class'              : 'all',
        'num_frames'         : total_raw['num_frames'],
        'num_objects'        : total_raw['num_objects'],
        'num_detections'     : total_raw['num_detections'],
        'num_misses'         : total_raw['num_misses'],
        'num_false_positives': total_raw['num_false_positives'],
        'num_switches'       : total_raw['num_switches'],
        'mota'               : total_mota,
        'motp'               : total_motp,
        'idf1'               : total_idf1,
    }], index=['All'])

    # Print results
    for class_df in class_dfs:
        summary_df = pd.concat([summary_df, class_df], ignore_index=True)
    
    # Add 'sum_overlap' and 'num_matches'
    summary_df['sum_overlap'] = sum_overlap_col
    summary_df['num_matches'] = num_matches_col

    # List of expected classes
    required_classes = ['boat', 'swimmer']

    # Fill zeros for missing classes
    for cls in required_classes:
        if cls not in summary_df['class'].values:
            empty_row = {col: 0 for col in summary_df.columns}
            empty_row['class'] = cls
            summary_df = pd.concat([summary_df, pd.DataFrame([empty_row])], ignore_index=True)

    # Optional: sort back to a consistent order
    summary_df = summary_df.sort_values(by='class')

    # Save to csv
    summary_df.to_csv(metrics_path, index=False)  # Save with the index

    return summary_df

def evaluate_tracking_fold(results_dfs, fold, fold_metrics_path):
    
    combined = pd.concat(results_dfs)

    # Group by class and sum the raw counts
    grouped = combined.groupby('class').agg({
        'num_frames': 'sum',
        'num_objects': 'sum',
        'num_detections': 'sum',
        'num_misses': 'sum',
        'num_false_positives': 'sum',
        'num_switches': 'sum',
        'sum_overlap': 'sum',
        'num_matches': 'sum'
    })

    # Compute MOTA, MOTP, IDF1
    grouped['mota'] = 1 - (
        (grouped['num_misses'] + grouped['num_false_positives'] + grouped['num_switches']) /
        grouped['num_objects'].replace(0, np.nan)
    )

    grouped['motp'] = grouped['sum_overlap'] / grouped['num_matches'].replace(0, np.nan)

    tp = grouped['num_detections']
    fp = grouped['num_false_positives']
    fn = grouped['num_misses']
    denom = (2 * tp + fp + fn).replace(0, np.nan)
    grouped['idf1'] = (2 * tp) / denom

    # Optional: round for display
    grouped = grouped.round(6)

    grouped.insert(0,  'fold', fold[4:])

    grouped.to_csv(fold_metrics_path, index=False)

    return grouped

def evaluate_tracking_deepsort(gt_path, pred_path, metrics_path, size=(640, 485)):
    """
    Evaluate DeepSORT tracking results using motmetrics.
    
    :param gt_file: Path to ground truth CSV file.
    :param pred_file: Path to predicted tracking CSV file.
    """

    # Load files to lists of data-frames 
    gt_df   = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)


    gt_df["x"] = gt_df["x"] / size[0]
    gt_df["y"] = gt_df["y"] / size[1]
    gt_df["w"] = gt_df["w"] / size[0]
    gt_df["h"] = gt_df["h"] / size[1]

    pred_df["x"] = pred_df["x"] / size[0]
    pred_df["y"] = pred_df["y"] / size[1]
    pred_df["w"] = pred_df["w"] / size[0]
    pred_df["h"] = pred_df["h"] / size[1]

    # Compute metrics independent of class (deepsort does not attribute a class_id)


    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(gt_df['frame_id'].unique())

    for frame in frames:
        gt_frame = gt_df[gt_df['frame_id'] == frame]
        pred_frame = pred_df[pred_df['frame_id'] == frame]


        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        pred_boxes = pred_frame[['x', 'y', 'w', 'h']].values

        gt_ids = gt_frame['track_id'].tolist()
        pred_ids = pred_frame['track_id'].tolist()

        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        acc.update(gt_ids, pred_ids, dist)

    metrics = [
        'num_frames', 'num_objects', 'num_detections',
        'num_misses', 'num_false_positives', 'num_switches',
        'mota', 'motp', 'idf1'
    ]

    mh = mm.metrics.create()
    summary_df = mh.compute(acc, metrics=metrics, name="all")

    # Save to csv
    summary_df.to_csv(metrics_path, index=True)  # Save with the index

    return summary_df

def create_hyperparameters_df(parent_dir, do_cv):

    split = "train" if do_cv else "test"

    # Regex to extract val1 and val2 from folder names like ntt_020_mt_050
    pattern = re.compile(r"ntt_(\d+)_mt_(\d+)")

    # To store all final data
    all_dfs = []

    # Loop through each folder in the parent directory
    for config_folder in os.listdir(parent_dir):
        match = pattern.match(config_folder)
        if not match:
            continue  # Skip non-matching folders

        val1_str, val2_str = match.groups()
        new_track_thresh = float(val1_str) / 100  # Convert "020" -> 0.2
        match_thresh = float(val2_str) / 100      # Convert "050" -> 0.5

        config_path = os.path.join(parent_dir, config_folder)
        folds = [f"fold{i}" for i in range(6)]
        fold_dfs = []

        # For each fold inside the current configuration
        for fold in folds:
            csv_path = os.path.join(config_path, "metrics", fold, f"tracking_metrics_{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                fold_dfs.append(df)
            else:
                print(f"Missing: {csv_path}")

        if not fold_dfs:
            continue  # Skip if nothing was found

        # Combine the folds for this configuration
        config_df = pd.concat(fold_dfs, ignore_index=True)

        # Add threshold info
        config_df.insert(0, "new_track_thresh", new_track_thresh)
        config_df.insert(1, "match_thresh", match_thresh)

        # Save intermediate CSV (optional)
        config_df.to_csv(os.path.join(config_path, "metrics/tracking_metrics_folds.csv"), index=False)

        # Add to final list
        all_dfs.append(config_df)

    # Combine all configurations into final dataframe
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by new_track_thresh and match_thresh
    final_df = final_df.sort_values(by=["new_track_thresh", "match_thresh"]).reset_index(drop=True)

    # Save final result
    final_df.to_csv(f"{parent_dir}/experiment_tracking_metrics.csv", index=False)

    print(f"✅ Done! Saved to {parent_dir}/experiment_tracking_metrics.csv\n")