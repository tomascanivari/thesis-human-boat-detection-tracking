import re
import os
import sys
import cv2
import csv
import torch
import shutil
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

# Import from custom utils/
from utils import evaluation_utils as ev

# Import deepsort
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deep_sort.deep_sort_app import run

# Clear memory
torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tracker on a sequence.")
    
    # Required arguments
    
    parser.add_argument('--tracker', type=str, required=True, help="Name of the tracker (botsort, bytetrack or deepsort)")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (CoastlineDrone-MOT, SeaDroneSee-MOT or Merged-MOT.)")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment (if it already exists all data is overwritten).")

    # Optional arguments
    parser.add_argument('--seq_name', type=str, required=False, help="Name of the sequence.", default="NOSEQ")
    parser.add_argument('--model_opt', type=int, required=False, help="Model options: 0 - Both (w/ cls); 1 - Coastline (/wo cls); 2 - Open-water (/wo cls)", default=0)
    parser.add_argument('--frame_width', type=int, required=False, help="Frame width.", default=640)
    parser.add_argument('--frame_height', type=int, required=False, help="Frame height.", default=485)
    parser.add_argument('--fps', type=int, required=False, help="Frames Per Second.", default=30)
    parser.add_argument('--save_video', action='store_true', help="Flag to save video.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()

    #####################################
    # --- Step 1: Process Arguments --- #
    #####################################

    fps = args.fps
    tracker = args.tracker
    dataset = args.dataset
    seq_name = args.seq_name
    exp_name = args.exp_name
    model_opt = args.model_opt
    save_video = args.save_video
    frame_width = args.frame_width
    frame_height = args.frame_height
    
    save_csv = True

    if seq_name == "NOSEQ":
        seq_name = ""
    
    ################################################
    # --- Step 2: Verify Directories Integrity --- #
    ################################################
    
    # Main folders
    exp_folder       = Path(f'runs/classify_detection_tracking/{exp_name}') # Root folder of experiment
    gts_folder       = Path(f'datasets/{dataset}/ground_truths')            # Root folder of ground-truths
    sequences_folder = Path(f'datasets/{dataset}/sequences/')               # Root folder of sequences
    
    # Experiment sub-folders
    videos_folder      = exp_folder / "videos"      # Videos folder
    metrics_folder     = exp_folder / "metrics"     # Metrics folder
    predictions_folder = exp_folder / "predictions" # Predictions folder

    gt_path = gts_folder / f"{seq_name}.csv"
    det_summary_path = exp_folder / "detection_metrics_all.csv"
    trk_summary_path = exp_folder / "tracking_metrics_all.csv"

    # Create and/or clear the experiment folder
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Create sub-folders
    videos_folder.mkdir(parents=True, exist_ok=True)
    metrics_folder.mkdir(parents=True, exist_ok=True)
    predictions_folder.mkdir(parents=True, exist_ok=True)

    ######################################################
    # --- Step 3: Show and Store Experiment Settings --- #
    ######################################################
    
    file = open(str(exp_folder / "settings.txt"), 'w')

    mopt2str = {0:"classification active and both models", 
                1:"classification not active and only coastline model", 
                2:"classification not active and only open-water model",
                3:"classification not active and only merged model"}
    
    for f in [sys.stdout, file]:
        if f == sys.stdout:
            start = "\033[1m"
            end = "\033[0m"
        elif f == file:
            start, end = "", ""
        print(f"{start}EXPERIMENT {exp_name} SETTINGS:{end}", file=f)
        print(f"           tracker = {start}{tracker}{end}", file=f)
        print(f"           dataset = {start}{dataset}{end}", file=f)
        print(f"         model_opt = {start}{mopt2str[model_opt]}{end}", file=f)
        print(f"        exp_folder = {start}{exp_folder}{end}", file=f)
        print(f"        gts_folder = {start}{gts_folder}{end}", file=f)
        print(f"       seqs_folder = {start}{sequences_folder}{end}", file=f)
        print(f"     videos_folder = {start}{videos_folder}{end}", file=f)
        print(f"    metrics_folder = {start}{metrics_folder}{end}", file=f)
        print(f"predictions_folder = {start}{predictions_folder}{end}", file=f)

    print("\n\n\033[1mSEQUENCES\033[0m")

    # Load all images, gts, predictions, videos, detection_metrics and tracking_metrics paths (per sequence)
    images_paths = []   # List os image_paths by sequence to be processed
    gt_paths = []
    predictions_paths = []
    video_paths = []
    detection_metrics_paths = []
    tracking_metrics_paths = []
    if seq_name == "":
        for sequence_folder in sorted(sequences_folder.iterdir()):
            if sequence_folder.is_dir():
                print("\t" + f"{sequence_folder}")
                image_path = sequence_folder / "img1-resized"
                images_paths.append(list(sorted(image_path.glob('*.jpg'))))
                gt_paths.append(gts_folder / f"{sequence_folder.name}.csv")
                predictions_paths.append(predictions_folder / f"tracking_predictions_{sequence_folder.name}.csv")
                video_paths.append(videos_folder / f"video_{sequence_folder.name}.avi")
                detection_metrics_paths.append(metrics_folder / f"detection_metrics_{sequence_folder.name}.csv")
                tracking_metrics_paths.append(metrics_folder / f"tracking_metrics_{sequence_folder.name}.csv")
    else:
        image_path = sequences_folder / f"{seq_name}/img1-resized"
        print(f"\t{sequences_folder}/{seq_name}")
        images_paths.append(sorted(list(image_path.glob('*.jpg'))))
        gt_paths.append(gts_folder / f"{seq_name}.csv")
        predictions_paths.append(predictions_folder / f"tracking_predictions_{seq_name}.csv")
        video_paths.append(videos_folder / f"video_{seq_name}.avi")
        detection_metrics_paths.append(metrics_folder / f"detection_metrics_{seq_name}.csv")
        tracking_metrics_paths.append(metrics_folder / f"tracking_metrics_{seq_name}.csv")



    #################################
    # --- Step 4: Main Pipeline --- #
    #################################
    det_metrics = []
    trk_metrics = []
    # For every video sequence run the pipeline

    for images_path, gt_path, predictions_path, video_path, detection_metrics_path, tracking_metrics_path in zip(images_paths, gt_paths, predictions_paths, video_paths,  detection_metrics_paths, tracking_metrics_paths):

        # Restart models for each video sequence
        do_cls = True   
        detector_no_cls : YOLO

        classifier         = YOLO('models/yolo11n_MergedDataset_CLS.pt')
        detector_merged    = YOLO('models/yolo12s_Merged.pt')
        detector_coastline = YOLO('models/yolo12s_CoastlineDrone.pt')
        detector_openwater = YOLO('models/yolo12m_SeaDroneSee.pt')
            
        if model_opt == 1:
            # No classification and only Coastline model
            do_cls = False
            detector_no_cls = detector_coastline
        elif model_opt == 2:
            # No classification and only Open-water model
            do_cls = False
            detector_no_cls = detector_openwater
        elif model_opt == 3:
            # No classification and only Open-water model
            do_cls = False
            detector_no_cls = detector_merged

        # Setup video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4 format
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        
        # Setup prediction 
        if tracker != "deepsort":
            file = open(predictions_path, 'w', newline='')
            header = ["frame_id", "category_id", "track_id", "x", "y", "w", "h", "conf", "xx", "yy", "zz"]
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

        ##############################################
        # --- Step 4: Classify, Detect and Track --- #
        ##############################################

        print(f"\n\n\033[1mRUNNING SEQUENCE {str(gt_path.name)[0:-4]}\033[0m")
        count = [0, 0]  # Number of images per class
        detections = []
        ctr = 0
        for img_path in tqdm(images_path):

            img_name = img_path.stem
            if img_path.name.startswith("frame_"):
                img_name = int(img_path.stem[6:])

            # Read image and store width and height
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

            # Classification
            if do_cls:
                # Predict class (coastline or openwater)
                class_result = classifier.predict(img, imgsz=img_w, verbose=False)
                predicted_class = int(class_result[0].probs.top1)

                # Detection model
                detection_model = detector_coastline if predicted_class == 0 else detector_openwater
                count[predicted_class] += 1 
            else:
                # Detection model
                detection_model = detector_no_cls

            # DeepSORT
            if tracker == "deepsort":
                
                detection_result = detection_model(source=img_path, verbose=False)

                # Process the detections
                names_to_id = {'ignored' : 0, 'swimmer' : 1, 'boat' : 2}
                for result in detection_result:
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
                        detections.append([int(re.search(r'(\d+)', os.path.basename(result.path)).group(1)), names_to_id[name], x, y, w, h, 1, -1, -1, -1])
                                    
            # BotSORT or ByteTrack
            elif tracker == "botsort.yaml" or "bytetrack.yaml":

                results = detection_model.track(source=img_path,  persist=True, tracker=tracker, verbose=False)

                # Save the results 
                for r in results:

                    annotated_frame = r.plot()
                    
                    cv2.imshow(f"{detection_model} Tracking", annotated_frame)
                    
                    if save_video:
                        out.write(annotated_frame)

                    # Save csv MOTChallenge format {'frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', -1, -1, -1}
                    if save_csv:           
                        data = r.summary()

                        for item in data:
                            
                            ctr += 1
                            row = {
                                'frame_id': int(re.search(r'(\d+)', os.path.basename(r.path)).group(1)),                                      # frame_id of the frame  
                                'category_id': int(item.get('class', ''))  ,                                                            
                                'track_id': item.get('track_id', ''),                           # track_id of the object
                                'x': item['box'].get('x1', ''),                                 # x-axis top-left corner
                                'y': item['box'].get('y1', ''),                                 # y-axis top-left corner
                                'w' : item['box'].get('x2', '') - item['box'].get('x1', ''),    # width of the bbox
                                'h': item['box'].get('y2', '') - item['box'].get('y1', ''),     # height of the bbox
                                'conf': item.get('confidence', ''),                             # confidence of the detection
                                'xx' : -1,
                                'yy' : -1,
                                'zz' : -1
                            }
                            
                            writer.writerow(row)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
        
        # Print class statistics
        if do_cls:
            print("\n\033[1mCLASSIFICATION\033[0m")
            print(f"Number of Coastline  images: {count[0]}")
            print(f"Number of Open-water images: {count[1]}")


        # Run deepsort after detection
        if tracker == "deepsort":
            predictions_path_npy = predictions_path.with_suffix(".npy")
            np.save(file=predictions_path_npy, arr=detections)
            
            run(images_path[0].parents[1], predictions_path_npy, predictions_path,
                0.8, 0.3, 0, 
                0.2, 100, True)

        ############################
        # --- Step 5: Evaluate --- #
        ############################
        if tracker != "deepsort":

            # Close and flush the predictions csv file...
            file.close()

            predictions, targets = ev.tracking2detection(gt_path, predictions_path)

            # Evaluate detection
            det_summary = ev.evaluate_detection(predictions, targets, detection_metrics_path)
            det_metrics.append(det_summary)

            print(f"\n\033[1mDETECTION METRICS\033[0m")
            print(det_summary)

            # Evaluate tracking
            trk_summary = ev.evaluate_tracking(gt_path, predictions_path, tracking_metrics_path)
            trk_metrics.append(trk_summary)

            print(f"\n\033[1mTRACKING METRICS\033[0m")
            print(trk_summary)
        else: 
            # Evaluate tracking
            trk_summary = ev.evaluate_tracking_deepsort(gt_path, predictions_path, tracking_metrics_path)
            trk_metrics.append(trk_summary)

            print(f"\n\033[1mTRACKING METRICS\033[0m")
            print(trk_summary)
            

        # Print save paths
        print(f"\n\033[1mSAVE PATHS\033[0m")
        if save_video:
            out.release()
            print(f"Saved video to                \033[1m{video_path}\033[0m")

        print(f"Saved tracking predictions to \033[1m{predictions_path}\033[0m")
        print(f"Saved detection metrics to    \033[1m{detection_metrics_path}\033[0m")
        print(f"Saved tracking  metrics to    \033[1m{tracking_metrics_path}\033[0m")

        cv2.destroyAllWindows()

    # Join all evaluation summaries
    if tracker != "deepsort":
        det_summary_all = pd.concat(det_metrics)
        det_summary_all.to_csv(det_summary_path)

    trk_summary_all = pd.concat(trk_metrics)
    trk_summary_all.to_csv(trk_summary_path)