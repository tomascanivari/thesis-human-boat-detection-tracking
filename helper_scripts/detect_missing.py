import os
import re

import os
import re

def find_and_create_missing_frames(directory,  max_frame):
    # Normalize extension
    extension=".txt"
    ext = extension.lower().lstrip(".")
    pattern = re.compile(rf"frame_(\d{{6}})\.{ext}$")

    frame_numbers = []

    print(f"Scanning directory: {directory}")
    for filename in os.listdir(directory):
        if not filename.lower().endswith(f".{ext}"):
            continue
        match = pattern.match(filename)
        if match:
            frame_numbers.append(int(match.group(1)))
        else:
            print(f"Skipping non-matching file: {filename}")

    if not frame_numbers:
        print("No matching files found.")
        return []

    frame_numbers.sort()
    min_frame = 0
    expected = set(range(min_frame, max_frame + 1))
    actual = set(frame_numbers)
    missing = sorted(expected - actual)

    print(f"Found {len(frame_numbers)} files. Missing {len(missing)} frames.")
    if missing:
        print("Creating placeholder files for:")
        for num in missing:
            filename = f"frame_{num:06d}.{ext}"
            path = os.path.join(directory, filename)
            open(path, 'w').close()  # Create empty file
            print(f"  → {filename}")
    else:
        print("No missing frames.")

    return missing

# Example usage
video_names = ['video_00', 'video_01', 'video_02', 'video_03', 'video_04', 
               'video_05', 'video_06', 'video_07', 'video_08', 'video_09', 'video_10', 'video_11']
max_frames = [431, 960, 305, 208, 1989, 895, 1720, 5446, 1523, 454, 305, 239]

for (name, max_frame) in zip(video_names, max_frames):
    directory_path = f"{name}/labels/train"  # Replace with your directory
    find_and_create_missing_frames(directory_path, max_frame-1)  # Change extension if needed

print(sum(max_frames))