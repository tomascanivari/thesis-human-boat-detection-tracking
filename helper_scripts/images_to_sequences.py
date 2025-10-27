import os
from PIL import Image

def images_to_sequences(source_dir, target_dir):

    # Create the output directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Target image size
    target_size = (640, 485)

    n_images = 0
    # Process each .png file in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.png'):
            # Load image
            image_path = os.path.join(source_dir, filename)
            image = Image.open(image_path)

            # Flip horizontally
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Resize
            resized = flipped.resize(target_size)

            # Convert to RGB (in case original is RGBA) and save as JPG
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(target_dir, f"{base_name}.jpg")
            resized.convert("RGB").save(output_path, format="JPEG", quality=95)
            n_images += 1
            print(f"Image '{image_path}' done.")

    print(f"Done! All {n_images} images flipped, resized, and saved as JPG.")

def main():

    # Source Images Directories
    src_img_dirs = ['videos/video_00/images/train', 
                   'videos/video_01/images/train', 
                   'videos/video_02/images/train',
                   'videos/video_03/images/train', 
                   'videos/video_04/images/train', 
                   'videos/video_06/images/train',
                   'videos/video_06/images/train', 
                   'videos/video_07/images/train', 
                   'videos/video_08/images/train',
                   'videos/video_09/images/train', 
                   'videos/video_10/images/train',
                   'videos/video_11/images/train']

    # Target Images Directories
    tgt_img_dirs = ['sequences/seq00',
                 'sequences/seq01',
                 'sequences/seq02',
                 'sequences/seq03',
                 'sequences/seq04',
                 'sequences/seq05',
                 'sequences/seq06',
                 'sequences/seq07',
                 'sequences/seq08',
                 'sequences/seq09',
                 'sequences/seq10',
                 'sequences/seq11']

    for src_img_dir, tgt_img_dir in zip(src_img_dirs, tgt_img_dirs):
        images_to_sequences(src_img_dir, tgt_img_dir)

if __name__ == "__main__":
    main()