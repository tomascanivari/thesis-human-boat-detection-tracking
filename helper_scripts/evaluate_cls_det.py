import os
import torch
import numpy as np
from ultralytics.utils.metrics import DetMetrics

# IoU calculation function (same as before)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to process multiple images
def process_multiple_images(gt_data, pred_data):
    """
    Process multiple images for evaluation, computing TP, Conf, Pred_cls, Target_cls.
    
    Args:
        gt_data (dict): Ground truth data for multiple images.
        pred_data (dict): Predicted data for multiple images.
        
    Returns:
        tuple: A tuple containing tensors for TP, Conf, Pred_cls, Target_cls.
    """
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []
    
    # IoU thresholds from 0.5 to 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Iterate over all images in the ground truth data
    for img_id in gt_data.keys():
        ground_truth = gt_data[img_id]
        predictions = pred_data[img_id]

        # Process each prediction
        for pred in predictions:
            pred_box = pred['bbox']         # [x1, y1, x2, y2]
            pred_conf = pred['confidence']  # Confidence
            pred_class = pred['class_id']   # Predicted class

            # Initialize TP for each IoU threshold as False initially (0)
            tp_for_prediction = np.zeros(len(iou_thresholds))

            # Flag to indicate if a match is found (True Positive)
            matched = False

            # Iterate over ground truth boxes
            for gt in ground_truth:
                gt_box = gt['bbox']         # [x1, y1, x2, y2]
                gt_class = gt['class_id']   # Ground truth class

                # Check if class matches and IoU threshold > 0.5
                if pred_class == gt_class:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= 0.5:
                        # print(f"Image {img_id} Predicted box: {pred_box} vs GT box: {gt_box} => IoU: {iou}")

                        # Update TP array for each threshold where IoU is greater
                        for i, threshold in enumerate(iou_thresholds):
                            if iou >= threshold:
                                tp_for_prediction[i] = 1  # Mark as TP for this IoU threshold
                        matched = True
                        break

            if not matched:
                tp_for_prediction[:] = 0  # All False (no match)

            # Append results
            tp_all.append(tp_for_prediction)
            conf_all.append(pred_conf)  # Confidence
            pred_cls_all.append(pred_class)  # Predicted class
            target_cls_all.append(gt_class if matched else -1)  # Target class or -1 if no match

    # Convert lists to tensors
    tp_all = torch.tensor(np.array(tp_all), dtype=torch.float32)
    conf_all = torch.tensor(conf_all, dtype=torch.float32)
    pred_cls_all = torch.tensor(pred_cls_all, dtype=torch.float32)
    target_cls_all = torch.tensor(target_cls_all, dtype=torch.float32)
    
    return tp_all, conf_all, pred_cls_all, target_cls_all

def cxcywh2xyxy(bbox, img_sz):
    
    # Unpack bbox and img_sz
    img_width, img_height = img_sz
    cxn, cyn, wn, hn = bbox
    
    # Convert from cx, cy, w, h (normalized) to x1, y1, x2, y2 (not normalized)

    x1 = (cxn - wn / 2) * img_width
    y1 = (cyn - hn / 2) * img_height
    x2 = (cxn + wn / 2) * img_width
    y2 = (cyn + hn / 2) * img_height

    return [x1, y1, x2, y2]

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
                    bbox = cxcywh2xyxy(list(map(float, parts[1:])), img_sz)
                    objects.append({'class_id': class_id, 'bbox': bbox})
                elif len(parts) == 6:
                    class_id = int(parts[0])
                    conf = float(parts[5])
                    bbox = cxcywh2xyxy(list(map(float, parts[1:5])), img_sz)
                    objects.append({'class_id': class_id, 'confidence': conf, 'bbox': bbox})
            annotations[img_name] = objects

    return annotations

img_sz = (640, 485)

gt_path   = "datasets/CoastlineDrone-OD/labels/val"
pred_path = "datasets/CoastlineDrone-OD/pred_labels/val"

gt_data   = read_txt_annotations(directory=gt_path, img_sz=img_sz)
pred_data = read_txt_annotations(directory=pred_path, img_sz=img_sz)

# Process multiple images
tp, conf, pred_cls, target_cls = process_multiple_images(gt_data, pred_data)



# Initialize the metrics object and process the results
metrics = DetMetrics()
metrics.process(tp, conf, pred_cls, target_cls)

# View the computed results
p, r, map50, map50_95 = [], [], [], []

for i in range(1,3):
    pi, ri, map50_i, map50_95_i = metrics.class_result(i)
    p.append(pi)
    r.append(ri)
    map50.append(map50_i)
    map50_95.append(map50_95_i)

print("Class \t Precision \t Recall \t mAP@50 \t mAP@50_95")

print(f"All\t {np.mean(p):.4f} \t {np.mean(r):.4f} \t {np.mean(map50):.4f} \t {np.mean(map50_95):.4f}")
for i in range(1,3):
    pi, ri, map50_i, map50_95_i = metrics.class_result(i)

    print(f"{i} \t {pi:.4f} \t {ri:.4f} \t {map50_i:.4f} \t {map50_95_i:.4f}")









































































# import torch
# from ultralytics.utils.metrics import DetMetrics

# import numpy as np


# # Define IoU function
# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     xx1, yy1, xx2, yy2 = box2

#     # Calculate the area of intersection
#     inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))

#     # Calculate the area of both boxes
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (xx2 - xx1) * (yy2 - yy1)

#     # Calculate the area of the union
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0

# # Input data: predictions and ground truth
# predictions = torch.tensor([[10, 10, 20, 20, 0.8, 1], [30, 30, 40, 40, 0.8, 1], [50, 50, 60, 60, 0.8, 2]])
# ground_truth = torch.tensor([[10, 10, 20, 20, 1], [30, 30, 40, 40, 1], [51, 52, 60, 60, 2]])

# # IoU thresholds from 0.5 to 0.95
# iou_thresholds = np.arange(0.5, 1.0, 0.05)

# # Initialize lists for TP, conf, pred_cls, target_cls
# tp = []
# conf = []
# pred_cls = []
# target_cls = []

# # Process each prediction
# for pred in predictions:
#     pred_box = pred[:4].numpy()  # [x1, y1, x2, y2]
#     pred_conf = pred[4].item()  # Confidence
#     pred_class = pred[5].item()  # Predicted class

#     # Initialize TP for each IoU threshold as False initially (0)
#     tp_for_prediction = np.zeros(len(iou_thresholds))

#     # Flag to indicate if a match is found (True Positive)
#     matched = False

#     # Iterate over ground truth boxes
#     for gt in ground_truth:
#         gt_box = gt[:4].numpy()  # [x1, y1, x2, y2]
#         gt_class = gt[4].item()  # Ground truth class

#         # Check if class matches and IoU threshold > 0.5
#         if pred_class == gt_class:
#             iou = calculate_iou(pred_box, gt_box)
            
#             if iou >= 0.5:
                
#                 print(f"Predicted box: {pred_box} vs GT box: {gt_box} => IoU: {iou}")

#                 # Update TP array for each threshold where IoU is greater
#                 for i, threshold in enumerate(iou_thresholds):
#                     if iou >= threshold:
#                         tp_for_prediction[i] = 1  # Mark as TP for this IoU threshold
#                 matched = True
#                 break

#     if not matched:
#         tp_for_prediction[:] = 0  # All False (no match)

#     # Append results
#     tp.append(tp_for_prediction)
#     conf.append(pred_conf)  # Confidence
#     pred_cls.append(pred_class)  # Predicted class
#     target_cls.append(gt_class if matched else -1)  # Target class or -1 if no match

# # Convert lists to tensors
# tp = torch.tensor(np.array(tp))
# conf = torch.tensor(conf)
# pred_cls = torch.tensor(pred_cls)
# target_cls = torch.tensor(target_cls)

# # Print shapes to verify
# print("TP shape:", tp.shape)  # Should be (N_preds, 10)
# print("Confidence shape:", conf.shape)
# print("Predicted class shape:", pred_cls.shape)
# print("Target class shape:", target_cls.shape)

# # Initialize metrics
# metrics = DetMetrics()

# # Process the metrics
# metrics.process(tp, conf, pred_cls, target_cls)

# # Compute final metrics
# print("Class \t Precision \t Recall \t mAP@50 \t mAP@50_95")
# for i in range(2):
#     pi, ri, map50_i, map50_95_i = metrics.class_result(i)

#     print(f"Class {i}\t {pi:.4f} \t {ri:.4f} \t {map50_i:.4f} \t {map50_95_i:.4f}")