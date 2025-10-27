import argparse
import pandas as pd
import motmetrics as mm

def load_mot_format(file_path, image_size):
    """
    Load MOT Challenge format tracking file.
    Expected columns: ["frame_id", "track_id", "x", "y", "w", "h", "conf", "xx", "yy", "zz"]
    Parameter image_size scales detections to 0-1 range.
    """
    df = pd.read_csv(file_path)

    df["x"] = df["x"] / image_size[0]
    df["y"] = df["y"] / image_size[1]
    df["w"] = df["w"] / image_size[0]
    df["h"] = df["h"] / image_size[1]

    return df[["frame_id", "track_id", "x", "y", "w", "h"]]

def evaluate_tracking(gt_file, pred_file, results_file, size):
    """
    Evaluate DeepSORT tracking results using motmetrics.
    
    :param gt_file: Path to ground truth CSV file.
    :param pred_file: Path to predicted tracking CSV file.
    """
    # Load files
    gt_df = load_mot_format(gt_file, size)              # Ground truth annotation image size (4K)
    pred_df = load_mot_format(pred_file, (640, 485))    # Predictions obtained from resized images 

    # Create MOT accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    frames = sorted(gt_df["frame_id"].unique())
    for frame in frames:
        # Get ground truth and prediction for the current frame
        gt_frame = gt_df[gt_df["frame_id"] == frame]
        pred_frame = pred_df[pred_df["frame_id"] == frame]

        gt_ids = list(gt_frame["track_id"])
        pred_ids = list(pred_frame["track_id"])
        
        # Compute IoU distance matrix
        distances = mm.distances.iou_matrix(gt_frame[["x", "y", "w", "h"]].values,
                                            pred_frame[["x", "y", "w", "h"]].values,
                                            max_iou=0.5)  # IoU threshold
        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_objects', 'num_detections', 
                                       'num_misses', 'num_false_positives', 'num_switches'], 
                         name="Tracking_Eval")

    print(summary)

    # Print results
    with open(f'{results_file}.csv', 'w') as file:
        file.write(summary.to_csv())
        print(f"Saved summary to {results_file}.csv")

    with open(f'{results_file}.txt', 'w') as file:
        file.write(summary.to_string())
        print(f"Saved summary to {results_file}.txt")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate Tracking Performance")
    parser.add_argument(
        "--gt_file", help="Ground-truth file.",
        required=True)
    parser.add_argument(
        "--pred_file", help="Tracker prediticions file.",
        required=True)
    parser.add_argument(
        "--results_file", help="MOT results file (save summary).",
        required=True)
    parser.add_argument(
        "--dataset", help="Dataset (CoastlineDrone-MOT or SeaDroneSee-MOT).",
        required=True)
    return parser.parse_args()

# Example usage
args = parse_args()

size = (640, 485)
if args.dataset == "SeaDroneSee-MOT":
    size = (3840, 2160)

evaluate_tracking(f"datasets/{args.dataset}/ground_truths/{args.gt_file}", f"results/{args.dataset}/{args.pred_file}", f"evaluation/{args.dataset}/{args.results_file}", size)