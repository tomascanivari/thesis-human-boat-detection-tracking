from pathlib import Path

DATASET_FOLDER                 = Path("CoastlineV2Dataset")

import cv2
import csv
import os
from collections import defaultdict

# --- Paths and settings ---
frames_folder = DATASET_FOLDER / "images"
frame_format = 'frame_{:06d}.jpg'  # Adjust if needed (e.g., {:06d})
fps = 30

def create_video(csv_file, output_video):
    # --- Read annotations ---
    annotations = defaultdict(list)  # {frame_id: [rows]}
    with open(csv_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            frame_id = int(row['frame_id'])
            x = int(float(row['x']))
            y = int(float(row['y']))
            w = int(float(row['w']))
            h = int(float(row['h']))
            track_id = row['track_id']
            category_id = row['category_id']
            annotations[frame_id].append((x, y, w, h, track_id, category_id))

    # --- Get image size from first frame ---
    first_frame = cv2.imread(os.path.join(frames_folder, frame_format.format(0)))
    height, width = first_frame.shape[:2]

    # --- Setup video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f"videos/{output_video}", fourcc, fps, (width, height))

    # --- Process and annotate each frame ---

    category_name = ['swimmer', 'boat']

    frame_ids = sorted(annotations.keys())
    for frame_id in frame_ids:
        frame_path = os.path.join(frames_folder, frame_format.format(frame_id))
        
        image = cv2.imread(frame_path)

        for x, y, w, h, track_id, category_id in annotations.get(frame_id, []):
            category_id = int(category_id)-1
            
            color = (0, 255, 0) if category_id == 0 else (255, 0, 255)
            # Draw box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

            # Add label (track_id and category)
            label = f"ID:{track_id} C:{category_name[category_id]}"
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.2, color, 1)

        video_writer.write(image)
        frame_id += 1

    video_writer.release()
    print("✅ Video saved:", f"videos/{output_video}")

if __name__ == "__main__":
    root_dir = DATASET_FOLDER / "tracking_labels"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:

            full_path = os.path.join(dirpath, filename)
            create_video(csv_file=full_path, output_video=f"{filename[0:-4]}.mp4")