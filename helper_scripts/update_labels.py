import os
import shutil

def copy_and_rename_txt_files(source_dir, dest_dir, start_number):
    
    n_labels = 0

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of .txt files and sort them
    txt_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".txt")])
    
    for i, filename in enumerate(txt_files):
        source_path = os.path.join(source_dir, filename)
        
        # Generate new filename
        new_number = start_number + i
        new_filename = f"frame_{new_number:06d}.txt"
        dest_path = os.path.join(dest_dir, new_filename)

        # Read, trim, and write content
        with open(source_path, 'r') as src_file, open(dest_path, 'w') as dst_file:
            for line in src_file:
                columns = line.strip().split()
                if len(columns) >= 6:
                    columns[0] = str(int(columns[0]) + 1)
                    trimmed_line = ' '.join(columns[:5])
                    dst_file.write(trimmed_line + '\n')
        n_labels += 1

    total_labels = start_number + n_labels
    print(f"src = {source_dir} ; Updated {n_labels} labels. Total is {total_labels}")
    
    return total_labels


# Example usage
video_names = ['video_00', 'video_01', 'video_02', 'video_03', 'video_04', 'video_05',
               'video_06', 'video_07', 'video_08', 'video_09', 'video_10', 'video_11']
destination_directory = "labels"
total_labels = 0

for video_name in video_names:
    source_directory = f"{video_name}/labels/train"
    total_labels = copy_and_rename_txt_files(source_directory, destination_directory, total_labels)