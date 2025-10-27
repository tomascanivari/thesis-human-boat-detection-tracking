import torch
from ensemble_boxes import weighted_boxes_fusion
import numpy as np

# --- Helpers to convert boxes ---
def xyxy_to_xywhn(boxes, img_width, img_height):
    boxes = boxes.cpu().numpy()
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return [[
        cx[i]/img_width,
        cy[i]/img_height,
        w[i]/img_width,
        h[i]/img_height
    ] for i in range(len(boxes))]

def xywhn_to_xyxy(boxes, img_width, img_height):
    boxes_abs = []
    for box in boxes:
        cx, cy, w, h = box
        x1 = (cx - w/2) * img_width
        y1 = (cy - h/2) * img_height
        x2 = (cx + w/2) * img_width
        y2 = (cy + h/2) * img_height
        boxes_abs.append([x1,y1,x2,y2])
    return torch.tensor(boxes_abs, dtype=torch.float32)

def empty_pred(image_id):
    return {'boxes': torch.empty((0,4)), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.int64), 'image_id': image_id}

def fuse_two_models_preds(p1, p2, img_width, img_height, iou_thr=0.3, skip_box_thr=0.0):
   
    boxes1 = p1['boxes'].cpu().numpy()
    scores1 = p1['scores'].cpu().numpy()
    labels1 = p1['labels'].cpu().numpy()

    boxes2 = p2['boxes'].cpu().numpy()
    scores2 = p2['scores'].cpu().numpy()
    labels2 = p2['labels'].cpu().numpy()

    # Normalize boxes to [0,1]
    boxes1[:, [0, 2]] /= img_width
    boxes1[:, [1, 3]] /= img_height
    boxes2[:, [0, 2]] /= img_width
    boxes2[:, [1, 3]] /= img_height

    boxes = [boxes1.tolist(), boxes2.tolist()]
    scores = [scores1.tolist(), scores2.tolist()]
    labels = [labels1.tolist(), labels2.tolist()]

    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        boxes, scores, labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        weights=[1, 2],  # Give more weight to the second model (small)
    )

    # Convert back to pixel coordinates (xyxy)
    boxes_fused = np.array(boxes_fused)
    boxes_fused[:, [0, 2]] *= img_width
    boxes_fused[:, [1, 3]] *= img_height

    # Prepare output in your original format, torch tensors
    fused_pred = {
        'boxes': torch.tensor(boxes_fused, dtype=torch.float32),
        'scores': torch.tensor(scores_fused, dtype=torch.float32),
        'labels': torch.tensor(labels_fused, dtype=torch.int64),
        'image_id': p1['image_id']  # assuming same image id
    }

    return fused_pred
# --- Main fusion loop ---
def fuse_datasets_preds(preds_model1, preds_model2, img_width=640, img_height=485, iou_thr=0.5, skip_box_thr=0.0):
    # Map image_id to preds
    dict1 = {p['image_id']: p for p in preds_model1}
    dict2 = {p['image_id']: p for p in preds_model2}

    all_image_ids = set(dict1.keys()) | set(dict2.keys())
    fused_preds = []

    for image_id in all_image_ids:
        p1 = dict1.get(image_id, {'boxes': torch.empty((0,4)), 'scores': torch.empty((0,)), 'labels': torch.empty((0,), dtype=torch.int64), 'image_id': image_id})
        p2 = dict2.get(image_id, {'boxes': torch.empty((0,4)), 'scores': torch.empty((0,)), 'labels': torch.empty((0,), dtype=torch.int64), 'image_id': image_id})

        fused = fuse_two_models_preds(p1, p2, img_width, img_height, iou_thr, skip_box_thr)
        fused_preds.append(fused)

    return fused_preds