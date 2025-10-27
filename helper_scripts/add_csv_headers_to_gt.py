import os
import pandas as pd

def add_header_to_csv(directory):
    # Define the new header
    header = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Process only CSV files
            file_path = os.path.join(directory, filename)
            
            # Read CSV file without headers
            df = pd.read_csv(file_path, header=None)
            
            # Assign new headers
            df.columns = header
            
            # Create new CSV filename
            new_filename = filename.replace('.txt', '.csv')
            new_file_path = os.path.join(directory, new_filename)
            
            # Save the updated CSV file
            df.to_csv(new_file_path, index=False)
            
            # Optionally, remove the original TXT file
            os.remove(file_path)
            
            print(f"Converted: {filename} -> {new_filename}")

# Usage
add_header_to_csv('datasets/SeaDroneSee-MOT/ground_truths/train/')
add_header_to_csv('datasets/SeaDroneSee-MOT/ground_truths/val/')