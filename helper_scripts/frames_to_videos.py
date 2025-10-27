import os
import json

def generate_video_frame_json(images_dir, video_lengths, output_json_path):
    """
    Generates a JSON mapping video IDs to lists of frame paths.

    Args:
        images_dir (str): Path to the directory containing frames named like 'frame_0', 'frame_1', etc.
        video_lengths (list of int): List where each element is the number of frames in a video.
        output_json_path (str): Path where the resulting JSON will be saved.
    """
    frame_index = 0
    video_frame_map = {}

    for i, length in enumerate(video_lengths):
        video_key = f"video_{i:02d}"
        frame_paths = []

        for j in range(length):
            frame_name = f"frame_{frame_index + j:06d}"
            frame_path = os.path.join(images_dir, frame_name)
            frame_paths.append(frame_path)

        video_frame_map[video_key] = frame_paths
        frame_index += length

    with open(output_json_path, 'w') as f:
        json.dump(video_frame_map, f, indent=4)

    print(f"JSON saved to {output_json_path}")

# Example usage:
if __name__ == "__main__":
    output_path      = "datasets/CoastlineDataset/frames_to_videos.json"
    images_directory = "datasets/CoastlineDataset/images"
    
    video_sizes = [431, 960, 305, 208, 1989, 895, 1720, 5446, 1523, 454, 305, 239]

    generate_video_frame_json(images_directory, video_sizes, output_path)