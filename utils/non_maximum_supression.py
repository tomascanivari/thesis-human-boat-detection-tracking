import torch

def combine_preds_with_nms_fast(preds1, preds2, iou_thresh=0.5, conf_thresh=0.10, device='cuda'):
    
    dict1 = {p['image_id']: p for p in preds1}
    dict2 = {p['image_id']: p for p in preds2}
    
    combined_preds = []
    all_image_ids = set(dict1.keys()).union(dict2.keys())
    
    for img_id in all_image_ids:
        p1 = dict1.get(img_id, None)
        p2 = dict2.get(img_id, None)

        def empty_pred():
            return {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'scores': torch.empty((0,), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
                'image_id': img_id
            }

        if p1 is None:
            p1 = empty_pred()
        if p2 is None:
            p2 = empty_pred()

        # Move to device and filter by conf threshold early
        boxes = torch.cat([p1['boxes'], p2['boxes']], dim=0).to(device)
        scores = torch.cat([p1['scores'], p2['scores']], dim=0).to(device)
        labels = torch.cat([p1['labels'], p2['labels']], dim=0).to(device)

        keep_mask = scores > conf_thresh
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        if boxes.numel() == 0:
            combined_preds.append(empty_pred())
            continue

        # Perform NMS class-wise using a vectorized approach
        # Offset boxes by class to do NMS in one go
        max_coord = boxes.max()
        offsets = labels.to(boxes.dtype) * (max_coord + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = torch.ops.torchvision.nms(boxes_for_nms, scores, iou_thresh)

        boxes = boxes[keep].cpu()
        scores = scores[keep].cpu()
        labels = labels[keep].cpu()

        combined_preds.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'image_id': img_id
        })

    return combined_preds