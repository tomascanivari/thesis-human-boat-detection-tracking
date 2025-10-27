import os
import pandas as pd

"""
Original labels from videos_00 to videos_11 to labels corresponding to horizontal flipped
images (and resized to 640, 485). YOLO Format (cx, cy, w, h) to MOTChallenge (left, top, w, h)
transformation is also applied.
"""

def labels_to_gt(labels_dir, csv_path):

    # Set your directory path here
    directory = labels_dir
    
    # Set image width and height
    image_width = 640
    image_height = 485

    # List to store all converted data
    data = []

    # Loop through all frame_*.txt files
    for filename in os.listdir(directory):
        if filename.startswith('frame_') and filename.endswith('.txt'):
            frame_number = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    category_id, cx, cy, w, h, track_id = parts
                    category_id = int(category_id) + 1
                    cx, cy, w, h = map(float, ( cx, cy, w, h))

                    # Flip horizontally
                    cx = 1.0 - cx

                    # Convert normalized to absolute pixel values
                    abs_w = w * image_width
                    abs_h = h * image_height
                    abs_cx = cx * image_width
                    abs_cy = cy * image_height

                    # Convert from center to top-left
                    left = abs_cx - abs_w / 2
                    top = abs_cy - abs_h / 2

                    data.append([frame_number, int(track_id), category_id, left, top, abs_w, abs_h, 1, -1, -1, -1])
                    
    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data, columns=['frame_id', 'track_id', 'category_id', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz'])

    # Sort by frame number track ID
    df = df.sort_values(by=['frame_id', 'track_id']).reset_index(drop=True)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    print(f"CSV file '{csv_path}' has been created successfully with {len(data)} instances!")

def main():

    # Labels Directories
    labels_dirs = ['videos/video_00/labels/train', 
                   'videos/video_01/labels/train', 
                   'videos/video_02/labels/train',
                   'videos/video_03/labels/train', 
                   'videos/video_04/labels/train', 
                   'videos/video_06/labels/train',
                   'videos/video_06/labels/train', 
                   'videos/video_07/labels/train', 
                   'videos/video_08/labels/train',
                   'videos/video_09/labels/train', 
                   'videos/video_10/labels/train',
                   'videos/video_11/labels/train']

    # CSV Paths
    csv_paths = ['ground_truths/seq00.csv',
                 'ground_truths/seq01.csv',
                 'ground_truths/seq02.csv',
                 'ground_truths/seq03.csv',
                 'ground_truths/seq04.csv',
                 'ground_truths/seq05.csv',
                 'ground_truths/seq06.csv',
                 'ground_truths/seq07.csv',
                 'ground_truths/seq08.csv',
                 'ground_truths/seq09.csv',
                 'ground_truths/seq10.csv',
                 'ground_truths/seq11.csv',]

    for labels_dir, csv_path in zip(labels_dirs, csv_paths):
        labels_to_gt(labels_dir, csv_path)

if __name__ == "__main__":
    main()