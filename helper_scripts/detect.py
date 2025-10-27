import re 
import os
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tracker on a sequence.")
    
    # Required arguments
    parser.add_argument('--seq_name', type=str, required=True, help="Name of the sequence.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    
    # Optional arguments
    parser.add_argument('--model', type=str, required=False, help="Path to the model.", default="models/yolo12s.pt")
    parser.add_argument('--width', type=int, required=False, help="Frame width.", default=640)
    parser.add_argument('--height', type=int, required=False, help="Frame height.", default=485)

    # Parse the arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()

    model = YOLO(args.model)

    image_dir = Path(f"datasets/{args.dataset}/sequences/{args.seq_name}/img1-resized/")

    results = model(source=image_dir, stream=True)

    frame_id = 0
    names_to_id = {'ignored' : 0, 'swimmer' : 1, 'boat' : 2, 'jetski' : 3, 'life_saving_appliances' : 4, 'buoy' : 5}

    detections = []
    for result in results:
        xyxy = result.boxes.xyxy    # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xywh = result.boxes.xywh    # center-x, center-y, width, height
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        for detection_info in zip(xyxy, xywh, names, confs):
            x = detection_info[0][0].item()
            y = detection_info[0][1].item()
            w = detection_info[1][2].item()
            h = detection_info[1][3].item()
            name = detection_info[2]
            conf = detection_info[3].item()
            # Save detection in MOTChallenge format (DeepSORT compatible)
            detections.append([int(re.search(r'(\d+)', os.path.basename(result.path)).group(1)), names_to_id[name], x, y, w, h, conf, -1, -1, -1])
            print([int(re.search(r'(\d+)', os.path.basename(result.path)).group(1)), names_to_id[name], x, y, w, h, conf, -1, -1, -1])
        frame_id += 1

    np.save(file=f"datasets/{args.dataset}/detections/{args.seq_name}.npy", arr=detections)
    print(f"Saved detections to \033[1m'datasets/{args.dataset}/detections/{args.seq_name}.npy'\033[0m")
