import os
import shutil

import tqdm

# List of folds you want to process, e.g. 0,1,2...
folds = [0, 1, 2, 3, 4, 5]  # add more folds here if needed

# Types of splits
splits = ['train', 'val', 'test']

# Dataset type keywords in paths and folder names
dataset_map = {
    'CoastlineDataset': 'coastline',
    'OpenWaterDataset': 'openwater'
}

for fold in folds:
    for split in splits:
        # Compose the filename to read
        txt_filename = f'datasets/MergedDataset/folds/{split}_fold{fold}.txt'
        
        if not os.path.isfile(txt_filename):
            print(f"File not found: {txt_filename}, skipping.")
            continue
        
        with open(txt_filename, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        
        for img_path in tqdm.tqdm(paths):
            # Determine dataset type from path string
            dataset_type = None
            for key in dataset_map:
                if key in img_path:
                    dataset_type = dataset_map[key]
                    break
            
            if dataset_type is None:
                print(f"Warning: Could not identify dataset type for {img_path}, skipping.")
                continue
            
            # Compose destination directory
            dest_dir = os.path.join(f'datasets/MergedDataset/classification/fold{fold}', split, dataset_type)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, f"{dataset_type}_{os.path.basename(img_path)}")

            # Copy file to destination folder
            try:
                shutil.copy(img_path, dest_path)
            except Exception as e:
                print(f"Error copying {img_path} to {dest_dir}: {e}")

print("Done.")