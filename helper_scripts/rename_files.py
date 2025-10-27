import os
import re
import argparse

def rename_files(directory):
    
    # Regular expression to match 6-digit filenames (e.g., 000001, 000010, 000123, 001000)
    pattern = re.compile(r'^\d{6}\..+$')

    # Check if any file already follows the 6-digit pattern
    has_6_digit_files = any(pattern.match(f) for f in os.listdir(directory))

    if has_6_digit_files:
        return

    # Get a sorted list of files in the directory
    files = sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]))

    # Rename each file with zero-padded numbers
    for filename in files:
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):  # Ensure it's a file
            name, ext = os.path.splitext(filename)
            new_name = f"{int(name):06d}{ext}"  # Convert to integer and pad with zeros
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tracker on a sequence.")
    
    # Required arguments
    parser.add_argument('--dir', type=str, required=True, help="Directory of the files to rename.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_arguments()

    rename_files(args.dir)