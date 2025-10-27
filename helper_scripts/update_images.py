import os
from PIL import Image

def convert_resize_and_rename_png_to_jpg(source_dir, dest_dir, start_number, target_size=(640, 485)):
    
    n_images = 0
    os.makedirs(dest_dir, exist_ok=True)

    # Get and sort all .png files
    png_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith('.png')])

    for i, filename in enumerate(png_files):
        source_path = os.path.join(source_dir, filename)
        new_number = start_number + i
        new_filename = f"frame_{new_number:06d}.jpg"
        dest_path = os.path.join(dest_dir, new_filename)

        # Open PNG, convert to RGB, resize, and save as JPG
        with Image.open(source_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(target_size)
            resized_img.save(dest_path, format='JPEG', quality=95)
            print(f"[{new_number}/14539] Saved {new_filename}.")
        n_images += 1

    total_images = start_number + n_images
    print(f"src = {source_dir} ; Updated {n_images} images. Total is {total_images}")
    return total_images

# Example usage
video_names = ['video_00', 'video_01', 'video_02', 'video_03', 'video_04', 'video_05',
               'video_06', 'video_07', 'video_08', 'video_09', 'video_10', 'video_11']
destination_directory = "images"
total_images = 0

for video_name in video_names:
    source_directory = f"{video_name}/images/train"
    total_images = convert_resize_and_rename_png_to_jpg(source_directory, destination_directory, total_images)