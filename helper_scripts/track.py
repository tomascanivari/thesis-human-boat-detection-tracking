import os
import re
import cv2
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tracker on a sequence.")
    
    # Required arguments
    parser.add_argument('--seq_name', type=str, required=True, help="Name of the sequence.")
    parser.add_argument('--tracker', type=str, required=True, help="Name of the tracker (bytetrack or botsort.)")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (CoastlineDrone-MOT or SeaDroneSee-MOT.)")

    # Optional arguments
    parser.add_argument('--model', type=str, required=False, help="Path to the model.", default="models/yolo12s.pt")
    parser.add_argument('--width', type=int, required=False, help="Frame width.", default=640)
    parser.add_argument('--height', type=int, required=False, help="Frame height.", default=485)
    parser.add_argument('--fps', type=int, required=False, help="Frames Per Second.", default=30)
    parser.add_argument('--save_video', action='store_true', help="Flag to save video.")
    parser.add_argument('--save_csv', action='store_true', help="Flag to save csv.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # Settings
    seq_name = args.seq_name
    dataset_name = args.dataset
    image_dir = Path(f"datasets/{dataset_name}/sequences/{args.seq_name}/img1-resized/")
    video_dir = Path(f"videos/{dataset_name}/{args.tracker[:-5]}")
    csv_dir  = Path(f"results/{dataset_name}/{args.tracker[:-5]}")

    tracker = args.tracker
    frame_width = args.width
    frame_height = args.height
    fps = args.fps
    save_video = args.save_video
    save_csv = args.save_csv
    video_path = video_dir / f"{seq_name}.avi"
    csv_path = csv_dir / f"{seq_name}.csv"

    # Create the directory if it doesn't exist
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"SETTINGS:")
    print(f"---------")
    print(f"images_path = {image_dir}")
    print(f"tracker = {tracker}")
    print(f"frame_width, frame_height, fps = ({frame_width}, {frame_height}, {fps})")
    print(f"save_video = {save_video}")
    if save_video:
        print(f"video_path = {video_path}")
    print(f"save_csv = {save_csv}")
    if save_csv:
        print(f"csv_path = {csv_path}")

    # Setup video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4 format
        out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    # Setup csv 
    if save_csv:
        file = open(csv_path, 'w')
        header = ["frame_id", "track_id", "x", "y", "w", "h", "conf", "xx", "yy", "zz"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

    # Load the YOLO model
    model = YOLO(args.model)

    # Tracking
    results = model.track(source=image_dir, stream=True, persist=True, tracker=tracker)

    for r in results:

        annotated_frame = r.plot()
        
        cv2.imshow(f"{model} Tracking", annotated_frame)
        
        if save_video:
            out.write(annotated_frame)

        if save_csv:           
            data = r.summary()

            for item in data:
                print(item)
                row = {
                    'frame_id': int(re.search(r'(\d+)', os.path.basename(r.path)).group(1)),
                    'track_id': item.get('track_id', ''),
                    'x': item['box'].get('x1', ''),
                    'y': item['box'].get('y1', ''),
                    'w' : item['box'].get('x2', '') - item['box'].get('x1', ''),
                    'h': item['box'].get('y2', '') - item['box'].get('y1', ''),
                    'conf': item.get('confidence', ''),
                    'xx' : -1,
                    'yy' : -1,
                    'zz' : -1
                }
                writer.writerow(row)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if save_video:
        out.release()
        print(f"Saved video to \033[1m{video_path}\033[0m")

    if save_csv:
        print(f"Saved results to \033[1m{csv_path}\033[0m")
        
    cv2.destroyAllWindows()