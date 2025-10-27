import os
import pandas as pd
import argparse


def merge_csv_files(folder_path, output_file):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Read and merge CSV files, adding a column with the filename
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df = df.iloc[:, 1:] 
        df.insert(0, "Filename", file)  # Insert filename as the first column
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # Save to a single file
    merged_df.to_csv(output_file, index=False)

    print(f"All CSV files merged into {output_file}")

if __name__ == "__main__":
    path = os.getcwd()
    
    set_names = [
        #"SeaDroneSee-MOT/bytetrack/train", 
        #"SeaDroneSee-MOT/bytetrack/val", 
        #"SeaDroneSee-MOT/botsort/train", 
        #"SeaDroneSee-MOT/botsort/val", 
        #"SeaDroneSee-MOT/deepsort/train", 
        #"SeaDroneSee-MOT/deepsort/val",
        "CoastlineDrone-MOT/bytetrack",
        "CoastlineDrone-MOT/botsort",
        "CoastlineDrone-MOT/deepsort",
        ]

    for set_name in set_names:
        folder_path = os.path.join(path, "evaluation", set_name)
        output_path = os.path.join(path, "evaluation", f"{set_name}.csv")

        merge_csv_files(folder_path, output_path)