import os

dirs = ["sequences/seq00/img1-resized",
        "sequences/seq01/img1-resized",
        "sequences/seq02/img1-resized",
        "sequences/seq03/img1-resized",
        "sequences/seq04/img1-resized",
        "sequences/seq05/img1-resized",
        "sequences/seq06/img1-resized",
        "sequences/seq07/img1-resized",
        "sequences/seq08/img1-resized",
        "sequences/seq09/img1-resized",
        "sequences/seq10/img1-resized",
        "sequences/seq11/img1-resized",]

for directory in dirs:

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith('frame_'):
            # New filename without 'frame_'
            new_name = filename.replace('frame_', '', 1)
            
            # Full paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_path, new_path)

    print("Renaming complete!")