import os
import io
import re
import sys
import cv2
import csv
import yaml
import json
import shutil
import copy
import argparse
import tempfile
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from utils import evaluation_utils as ev

class StdoutFilter(io.StringIO):
    def __init__(self, suppress_phrase):
        super().__init__()
        self.suppress_phrase = suppress_phrase

    def write(self, s):
        if self.suppress_phrase not in s:
            sys.__stdout__.write(s)  # write to real stdout if allowed

def load_sequences(seq_path):

    with open(seq_path, "r") as f:
        data = json.load(f)

    return data["folds"], data["videos"]

def load_model(models_dir, fold):
    
    return YOLO(model=f"{models_dir}/{fold}.pt")

def load_classification_model(fold, v2_flag):

    model_str = "ClassificationV2Models" if v2_flag else "ClassificationModels"

    return YOLO(model=f"models/{model_str}/{fold}.pt")

def load_tracker_config(tracker_path, cv_dict):
   
    tracker_configs = {}

    # Load original YAML config
    with open(tracker_path, 'r') as f:
        tracker_config = yaml.safe_load(f)

    for (cv_dir, cv_values) in cv_dict.items():
        
        updated_tracker_config = copy.deepcopy(tracker_config)

        updated_new_track_thresh, updated_match_thresh = cv_values

        # Override parameters programmatically
        updated_tracker_config['new_track_thresh'] = updated_new_track_thresh
        updated_tracker_config['match_thresh']     = updated_match_thresh

        tracker_configs[cv_dir] = updated_tracker_config

        print(cv_dir, "    ", f"new_track_thresh={tracker_configs[cv_dir]['new_track_thresh']:.2f}", "    ", f"match_thresh={tracker_configs[cv_dir]['match_thresh']:.2f}")

    return tracker_configs

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CLS+DET+TRK+EVL Pipeline on the TEST split of a dataset. \n"
                                                 "Also has a mode for CV that works on the TRAIN split.")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment (if it already exists all data is overwritten).")
    parser.add_argument('--tracker', type=str, required=True, help="Name of the tracker (botsort, bytetrack or deepsort)")

    # Optional
    parser.add_argument('--coastline_model', type=str, required=False, help="Specify the Coastline model", default="Coastline")
    parser.add_argument('--do_cv', action='store_true', help='If TRUE does CV on the TRAIN split. If FALSE default pipeline is used on TEST split.')
    parser.add_argument('--save_video', action='store_true', help='If TRUE saves video with the IDs and BBoxes.')

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():

    #####################################
    # --- Step 1: Process Arguments --- #
    #####################################
    args       = parse_arguments()
    dataset    = args.dataset           # Dataset name (assumes it has the correct structure and is in correct directory '/datasets/')
    exp_name   = args.exp_name          # Experiment directory name (/runs/tracking/{exp_name})
    tracker    = args.tracker           # Tracker file name (assumes it is in correct directory '/trackers/')
    do_cv      = args.do_cv             # Mode of operation (if TRUE does CV on TRAIN, else does EVL on TEST)
    save_video = args.save_video        # Flag to save the videos on the experiment directory
    cl_model   = args.coastline_model

    v2_flag = True if "V2" in cl_model else False

    ################################################
    # --- Step 2: Verify Directories Integrity --- #
    ################################################
    exp_folder   = Path(f'runs/tracking/{exp_name}')  
    dataset_dir  = Path(f"datasets/{dataset}Dataset")
    trackers_dir = Path(f"trackers/")
    labels_dir   = dataset_dir / "tracking_labels"
    seq_path     = dataset_dir / "sequences.json"
    tracker_path = trackers_dir / f"{tracker}.yaml"


    # Create and/or clear the experiment folder
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Prepare sub-directories for each mode
    videos_cv_dir      = []
    metrics_cv_dir     = []
    predictions_cv_dir = []
    
    if do_cv:  

        # Cross Validation sub-directories and respective parameter names and values
        cross_validation_dict = {exp_folder / "ntt_020_mt_050" : (0.20, 0.50), exp_folder / "ntt_020_mt_065" : (0.20, 0.65), exp_folder / "ntt_020_mt_080" : (0.20, 0.80), 
                                 exp_folder / "ntt_035_mt_050" : (0.35, 0.50), exp_folder / "ntt_035_mt_065" : (0.35, 0.65), exp_folder / "ntt_035_mt_080" : (0.35, 0.80),
                                 exp_folder / "ntt_050_mt_050" : (0.50, 0.50), exp_folder / "ntt_050_mt_065" : (0.50, 0.65), exp_folder / "ntt_050_mt_080" : (0.50, 0.80)}

        # Create each sub-directory and the necessary folders
        for cv_folder in cross_validation_dict.keys():
            cv_folder.mkdir(parents=True, exist_ok=True)

            # CV sub-folders
            videos_dir       = cv_folder / "videos"      # Videos folder
            metrics_dir      = cv_folder / "metrics"     # Metrics folder
            predictions_dir  = cv_folder / "predictions" # Predictions folder

            # Create sub-folders
            metrics_dir.mkdir(parents=True, exist_ok=True)
            predictions_dir.mkdir(parents=True, exist_ok=True)
            if save_video: 
                videos_dir.mkdir(parents=True, exist_ok=True)

            # Add to list
            videos_cv_dir.append(videos_dir)
            metrics_cv_dir.append(metrics_dir)
            predictions_cv_dir.append(predictions_dir)

    else:
        # Perform CLS+DET+TRK+EVL on TEST split
        best_pairs = {
            "Coastline"   : {
                "botsort"   : ("ntt_035_mt_080", (0.35, 0.8)),
                "bytetrack" :  ("ntt_035_mt_080", (0.35, 0.8))
            },
            "OpenWater"   : {
                "botsort"   : ("ntt_035_mt_080", (0.35, 0.8)),
                "bytetrack" : ("ntt_035_mt_080", (0.35, 0.8))
            },
            "Merged"      : {
                "botsort"   : ("ntt_035_mt_080", (0.35, 0.8)),
                "bytetrack" : ("ntt_035_mt_080", (0.35, 0.8))
            },
        }

        # Only 1 pair (best pair)
        cross_validation_dict = {exp_folder / best_pairs[dataset][tracker][0] : best_pairs[dataset][tracker][1]}
        
        cv_folder = exp_folder / best_pairs[dataset][tracker][0]
        cv_folder.mkdir(parents=True, exist_ok=True)

        # CV sub-folders
        videos_dir       = cv_folder / "videos"      # Videos folder
        metrics_dir      = cv_folder / "metrics"     # Metrics folder
        predictions_dir  = cv_folder / "predictions" # Predictions folder

        # Create sub-folders
        metrics_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        if save_video: 
            videos_dir.mkdir(parents=True, exist_ok=True)

        # Add to list
        videos_cv_dir.append(videos_dir)
        metrics_cv_dir.append(metrics_dir)
        predictions_cv_dir.append(predictions_dir)

    #############################################
    # --- Step 3: Load and Process Pipeline --- #
    #############################################

    # Load the sequences data (videos in each fold and frames in each video)
    folds_to_videos, videos_to_frames = load_sequences(seq_path)

    # Load all tracker configs
    tracker_configs = load_tracker_config(tracker_path, cross_validation_dict)

    # Process each tracker configuration (grid of hyperparameters)
    for cv_dir, videos_dir, metrics_dir, predictions_dir in tqdm(zip(cross_validation_dict.keys(), videos_cv_dir, metrics_cv_dir, predictions_cv_dir), desc= "GrdSch", total=len(videos_cv_dir)):
        
        # Create temporary yaml file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(tracker_configs[cv_dir], tmp_file)
            tmp_tracker_path = tmp_file.name

        # Process each fold
        for fold in tqdm(folds_to_videos.keys(), desc="  Fold", leave=False):  
            # Process each split 
            splits = ["train"] if do_cv else ["test"]
            for split in splits:
                
                # Create experiment sub-folders
                videos_fold_split_dir      = videos_dir / fold / split
                metrics_fold_split_dir     = metrics_dir / fold / split
                predictions_fold_split_dir = predictions_dir / fold / split

                # videos_fold_split_dir.mkdir(parents=True, exist_ok=True)
                videos_fold_split_dir.mkdir(parents=True, exist_ok=True)
                metrics_fold_split_dir.mkdir(parents=True, exist_ok=True)
                predictions_fold_split_dir.mkdir(parents=True, exist_ok=True)

                trk_metrics_split = []

                trk_metrics_split_path = metrics_dir / fold / f"tracking_metrics_{split}.csv"

                # Process each video of this fold and split 
                for video in tqdm(folds_to_videos[fold][split], desc=" Video", leave=False):

                    # Paths
                    labels_path      = labels_dir / f"{video}.csv"
                    video_path       = videos_fold_split_dir / f"{video}.avi"
                    predictions_path = predictions_fold_split_dir / f"tracking_predictions_{video}.csv"
                    trk_metrics_path = metrics_fold_split_dir / f"tracking_metrics_{video}.csv"

                    # Load the model before processing each video (to reset the tracking ids)
                    detection_model_ow = load_model(models_dir="models/OpenWaterModels", fold=fold)
                    detection_model_cl = load_model(models_dir=f"models/{cl_model}Models", fold=fold)

                    classification_model = load_classification_model(fold=fold, v2_flag=v2_flag)

                    # Setup video writer
                    if save_video:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4 format
                        out = cv2.VideoWriter(video_path, fourcc, 24, (640, 485))
                                          
                    # Setup prediction 
                    file = open(predictions_path, 'w', newline='')
                    header = ["frame_id", "category_id", "track_id", "x", "y", "w", "h", "conf", "xx", "yy", "zz"]
                    writer = csv.DictWriter(file, fieldnames=header)
                    writer.writeheader()
                    
                    ctr = 0
                    skip = False
                    for img_path in tqdm(videos_to_frames[video], desc=" Frame", leave=False):
                        
                            
                        img_path = Path(img_path)

                        # Read image and store width and height
                        img = cv2.imread(str(img_path))
                        img_h, img_w = img.shape[:2]

                        # Predict class (coastline or openwater)
                        class_result = classification_model.predict(img, imgsz=img_w, verbose=False)
                        predicted_class = int(class_result[0].probs.top1)

                        # predicted_class = 0 if "cl" in video else 1

                        detection_model = detection_model_cl if predicted_class == 0 else detection_model_ow

                        # # Text to suppress
                        # suppress_text = "not enough matching points"
                        # sys.stdout = StdoutFilter(suppress_text)

                        # BotSORT or ByteTrack
                        results = detection_model.track(source=img_path, persist=True, tracker=tmp_tracker_path, verbose=False)

                        # # Restore stdout afterward
                        # sys.stdout = sys.__stdout__

                        # Save the results 
                        for r in results:

                            annotated_frame = r.plot()
                            
                            cv2.imshow(f"{detection_model} Tracking", annotated_frame)
                            
                            # Save frame to the video
                            if save_video:
                                out.write(annotated_frame)

                            # Save csv MOTChallenge format {'frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', -1, -1, -1}
                            data = r.summary()

                            for item in data:
                                
                                ctr += 1
                                row = {
                                    'frame_id': int(re.search(r'(\d+)', os.path.basename(r.path)).group(1)),                                      # frame_id of the frame  
                                    'category_id': int(item.get('class', '')) ,                                                            
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
                    
                    if save_video:
                        out.release()

                    ############################
                    # --- Step 5: Evaluate --- #
                    ############################

                    # Close and flush the predictions csv file...
                    file.close()

                    # Evaluate tracking

                    if not skip:
                        trk_summary = ev.evaluate_tracking(labels_path, predictions_path, trk_metrics_path)
                        trk_summary = trk_summary.fillna(0)
                        trk_metrics_split.append(trk_summary)

                    # print(f"\n\033[1mTRACKING METRICS\033[0m")
                    # print(trk_summary)

                

                # Agreggate and compute metrics for split bases on individual videos metrics 
                final_df = ev.evaluate_tracking_fold(trk_metrics_split, fold, trk_metrics_split_path)
                final_df = final_df.fillna(0)
                final_df.to_csv(trk_metrics_split_path)

        os.remove(tmp_tracker_path)

    # Summarize hyperparameters results for this experience
    ev.create_hyperparameters_df(parent_dir=exp_folder, do_cv=do_cv)

#
#
#

# Save original warning method
original_warning = LOGGER.warning

# Supress WARNING message of not enough matching point when using BotSort
def custom_warning(msg, *args, **kwargs):
    if "not enough matching points" in str(msg):
        return  # suppress this warning
    return original_warning(msg, *args, **kwargs)

# Patch it
LOGGER.warning = custom_warning

# Call the main function
main()