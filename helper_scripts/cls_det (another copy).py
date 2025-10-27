import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ultralytics.models.yolo.detect.val import DetectionValidator

##########################################################################################
#                                                                                        #
#                                                                                        #
#                                 HELPER FUNCTIONS                                       #
#                                                                                        #
#                                                                                        #
##########################################################################################

# Function to compute Intersection-over-Union
def calculate_iou(pred_bbox, gt_bbox):
    x1, y1, x2, y2 = pred_bbox
    xx1, yy1, xx2, yy2 = gt_bbox
    inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to process multiple images annotations and obtain the TP, confidence, pred_cls
# and target_cls of all detections in all images.
def process_multiple_images(gt_data, pred_data, min_threshold = 0.5):
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
                    if iou >= min_threshold:
                        # Update TP array for each threshold where IoU is greater
                        for i, threshold in enumerate(iou_thresholds):
                            if iou >= threshold:
                                tp_for_prediction[i] = 1  # Mark as TP for this IoU threshold
                        matched = True
                        break

            if not matched:
                print("no match")
                tp_for_prediction[:] = 0  # All False (no match)

            # Append results
            tp_all.append(tp_for_prediction)                    # True Positives
            conf_all.append(pred_conf)                          # Confidence
            pred_cls_all.append(pred_class)                     # Predicted class
            target_cls_all.append(gt_class if matched else -1)  # Target class or -1 if no match

    # Convert lists to tensors
    tp_all = torch.tensor(np.array(tp_all), dtype=torch.float32)
    conf_all = torch.tensor(conf_all, dtype=torch.float32)
    pred_cls_all = torch.tensor(pred_cls_all, dtype=torch.float32)
    target_cls_all = torch.tensor(target_cls_all, dtype=torch.float32)
    
    return tp_all, conf_all, pred_cls_all, target_cls_all

# Function that converts bounding boxes between [cxn, cyn, wn, hn] to [x1, y1, x2, y2]
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

# Function that reads YOLO [class_id, (optional)conf, cxn, cyn, wn, hn] annotations from
# all txt files in a directory and stores them in a dictionary
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
                # Ground-truths
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = cxcywh2xyxy(list(map(float, parts[1:])), img_sz)
                    objects.append({'class_id': class_id, 'bbox': bbox})
                # Predictions
                elif len(parts) == 6:
                    class_id = int(parts[0])
                    conf = float(parts[5])
                    bbox = cxcywh2xyxy(list(map(float, parts[1:5])), img_sz)
                    objects.append({'class_id': class_id, 'confidence': conf, 'bbox': bbox})
            
            annotations[img_name] = objects

    return annotations

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tracker on a sequence.")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--split', type=str, required=True, help="Split of the dataset (train or val).")

    # Optional arguments
    parser.add_argument('--width', type=int, required=False, help="Frame width.", default=640)
    parser.add_argument('--height', type=int, required=False, help="Frame height.", default=485)

    # Parse the arguments
    args = parser.parse_args()
    
    return args

##########################################################################################
#                                                                                        #
#                                                                                        #
#                                      MAIN SCRIPT                                       #
#                                                                                        #
#                                                                                        #
##########################################################################################

# Settings
args    = parse_arguments()

split   = args.split
dataset = args.dataset
img_sz  = (args.width, args.height)


# Load models
classifier = YOLO('models/yolo11n_MergedDataset_CLS.pt')
detector_coastline = YOLO('models/yolo12s_CoastlineDrone.pt')
detector_openwater = YOLO('models/yolo12s_SeaDroneSee.pt')

# Paths
gt_folder     = Path(f'datasets/{dataset}/labels/{split}') 
pred_folder   = Path(f'datasets/{dataset}/pred_labels/{split}')
input_folder  = Path(f'datasets/{dataset}/images/{split}')
output_folder = Path(f'evaluation/{dataset}-CLS')
output_path   = output_folder / f'{split}.txt'
pred_folder.mkdir(parents=True, exist_ok=True)
output_folder.mkdir(parents=True, exist_ok=True)

# Get all image paths
image_paths = list(input_folder.glob('*.jpg'))

# Main pipeline
if False:
    annotation_id = 0
    count = [0, 0]
    for img_path in tqdm(image_paths):
        img_name = img_path.name
        img_stem = img_path.stem
        gt_path = gt_folder / f"{img_stem}.txt"
        pred_path = pred_folder / f"{img_stem}.txt"


        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        # Step 1: Classification
        class_result = classifier.predict(img, imgsz=640, verbose=False)
        predicted_class = int(class_result[0].probs.top1)
        count[predicted_class] += 1

        # Step 2: Detection
        detection_model = detector_coastline if predicted_class == 0 else detector_openwater
        detection_result = detection_model.predict(img, save=False, verbose=False)

        # Step 3: Save predictions
        with open(file=pred_path, mode="w") as f:
            preds_boxes = []
            boxes = detection_result[0].boxes.xywhn.cpu().numpy()
            confs = detection_result[0].boxes.conf.cpu().numpy()
            clsss = detection_result[0].boxes.cls.cpu().numpy()
            for box, conf, clss in zip(boxes, confs, clsss):
                preds_boxes.append([int(clss), box[0], box[1], box[2], box[3], conf])
            # Iterate through each inner list
            for row in preds_boxes:
                # Convert each list to a string and write it as a new row in the file
                f.write(' '.join(map(str, row)) + '\n')

    print(f"Coastline model  : {count[0]}\n"
        f"Open-water model : {count[1]}\n")

# Step 4: Evaluation
gt_data   = read_txt_annotations(directory=gt_folder, img_sz=img_sz)
pred_data = read_txt_annotations(directory=pred_folder, img_sz=img_sz)

metrics = DetMetrics(names={0: 'ignored', 1:'swimmer', 2:'boat'})
val = DetectionValidator()

tp_all, pred_conf_all, pred_cls_all, gt_cls_all = [], [], [], []

for img_id in gt_data.keys():
    ground_truth = gt_data[img_id]
    predictions  = pred_data[img_id]

    print(ground_truth)

    print(predictions)

    # Process each images prediction and GT separately.
    # We use the prediction from model. But it can be your saved predictions. Shape [N, 6]. Type: Tensor.
    preds = []
    for item in predictions:
        bbox = item['bbox']
        confidence = item['confidence']
        class_id = item['class_id']
        
        # Create a row with bbox, confidence, and class_id
        row = bbox + [confidence, class_id]
        preds.append(row)

    # Convert the list to a tensor (N, 6)
    boxes = torch.tensor(preds)
    if len(boxes) != 0:
        pred_conf = boxes[:, 4]
        pred_cls = boxes[:, 5]
    else:
        continue

    gts = []
    for item in ground_truth:
        bbox = item['bbox']
        class_id = item['class_id']
        
        # Create a row with bbox, confidence, and class_id
        row = bbox + [class_id]
        gts.append(row)

    # Convert the list to a tensor (N, 6)
    gts = torch.tensor(gts)
    if len(gts) != 0:
        gt_boxes = gts[:, :4]
        gt_cls = gts[:, 4]
    else: 
        continue

    
    tp = val._process_batch(boxes, gt_boxes, gt_cls).int()
    
    
    

    for tpp, pcnf, pcls in zip(tp, pred_conf, pred_cls):
        tp_all.append(tpp)
        pred_conf_all.append(pcnf)
        pred_cls_all.append(pcls)
        if torch.all(tpp == 0):
            gt_cls_all.append(-1)
        else:
            gt_cls_all.append(pcls)        

print(tp_all)
print(pred_conf_all)
print(pred_cls_all)
print(gt_cls_all)

tp_all = np.array(tp_all)
pred_conf_all = np.array(pred_conf_all)
pred_cls_all = np.array(pred_cls_all)
gt_cls_all = np.array(gt_cls_all)

print(np.shape(tp_all), np.shape(pred_conf_all), np.shape(pred_cls_all), np.shape(gt_cls_all))
metrics.process(tp_all, pred_conf_all, pred_cls_all, gt_cls_all)

# View the computed results
p, r, map50, map50_95 = [], [], [], []

for i in range(3):
    pi, ri, map50_i, map50_95_i = metrics.class_result(i)
    p.append(pi)
    r.append(ri)
    map50.append(map50_i)
    map50_95.append(map50_95_i)

print("Class \t Precision \t Recall \t mAP@50 \t mAP@50_95")

print(f"All\t {np.mean(p):.4f} \t {np.mean(r):.4f} \t {np.mean(map50):.4f} \t {np.mean(map50_95):.4f}")
for i in range(3):
    pi, ri, map50_i, map50_95_i = metrics.class_result(i)

    print(f"{i} \t {pi:.4f} \t {ri:.4f} \t {map50_i:.4f} \t {map50_95_i:.4f}")

with open(file=output_path, mode="w") as f:
    f.write("Class \t Precision \t Recall \t mAP@50 \t mAP@50_95\n")
    f.write(f"All\t {np.mean(p):.4f} \t {np.mean(r):.4f} \t {np.mean(map50):.4f} \t {np.mean(map50_95):.4f}\n")
    for i in range(2):
        pi, ri, map50_i, map50_95_i = metrics.class_result(i)

        f.write(f"{i} \t {pi:.4f} \t {ri:.4f} \t {map50_i:.4f} \t {map50_95_i:.4f}\n")