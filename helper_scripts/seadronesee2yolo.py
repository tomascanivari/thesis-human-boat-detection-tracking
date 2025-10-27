import os
import json

def convert_sds_to_yolo(sds_annotation_path, output_dir):
    """
    Convert SeaDroneSee Object Detection V2 dataset annotations (COCO format) to YOLO format.
    COCO format - JSON with bbox [x, y, width, height]
    YOLO format - class_id x_center y_center width height
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(sds_annotation_path, 'r') as f:
        sds_data = json.load(f)

    # Map categories to YOLO class IDs
    categories = {cat['id']: idx for idx, cat in enumerate(sds_data['categories'])}

    # Process annotations
    for img in sds_data['images']:
        img_id = img['id']
        img_name = img['file_name']
        img_width, img_height = img['width'], img['height']
        
        annotation_file = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(annotation_file, 'w') as f_out:
            for ann in sds_data['annotations']:
                if ann['image_id'] == img_id:
                    cat_id = ann['category_id']
                    bbox = ann['bbox']  # sds bbox format: [x_min, y_min, width, height]
                    
                    # Convert bbox to YOLO format
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # Write to file
                    f_out.write(f"{categories[cat_id]} {x_center} {y_center} {width} {height}\n")
                    print(f"Wrote [{categories[cat_id]} {x_center} {y_center} {width} {height}] to {annotation_file}")

# Example usage
sds_annotation_paths = ["datasets/SeaDroneSee/annotations/instances_train.json",
                        "datasets/SeaDroneSee/annotations/instances_val.json"]
output_dirs = ["datasets/SeaDroneSee/labels/train",
               "datasets/SeaDroneSee/labels/val"]

for ann_path, out_dir in zip(sds_annotation_paths, output_dirs):
    convert_sds_to_yolo(ann_path, out_dir)