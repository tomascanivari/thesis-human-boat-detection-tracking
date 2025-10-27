import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import t

GRID_SEARCH_METRICS_PATH = 'tracking_grid_search_metrics.csv'

DATASETS = ['Coastline', 'OpenWater', 'Merged']
TRACKERS = ['BotSort', 'ByteTrack']

def evaluate_hyperparameters(dataset, metrics):

    # Load CSV file
    df = pd.read_csv(GRID_SEARCH_METRICS_PATH)

    tracker_labels = {'BotSort': 'BoT-SORT', 'ByteTrack': 'ByteTrack'}

    # Filter only 'all' class
    df = df[(df['dataset'] == dataset) & (df["class"] == "all")]

    # Ensure output directory exists
    os.makedirs(f"plots/{dataset}", exist_ok=True)

    # Ensure thresholds are numeric
    df["new_track_thresh"] = pd.to_numeric(df["new_track_thresh"])
    df["match_thresh"] = pd.to_numeric(df["match_thresh"])

    metric_labels = {'mota': 'MOTA',
                     'motp': 'MOTP',
                     'idf1': r"$\text{F1-score}_{\text{trk}}$",
                     'num_switches': 'IDS'}

    for config, order in zip(["config1", "config2"], ["(ntt, mt)", "(mt, ntt)"]):

        if config == "config1":
            df["config1"] = df.apply(lambda row: f"ntt={row['new_track_thresh']}, mt={row['match_thresh']}", axis=1)
        elif config == "config2":
            # Sort by match_thresh first, then new_track_thresh
            df = df.sort_values(by=["match_thresh", "new_track_thresh"])
            df["config2"] = df.apply(lambda row: f"mt={row['match_thresh']}, ntt={row['new_track_thresh']}", axis=1)

        n_rows = len(metrics)
        n_cols = 1
        col = 0

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(12*n_cols, 4*n_rows), sharex="col"
        )

        # If only 1 row/col, make axes always 2D array
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for row, metric in enumerate(metrics):

            ax = axes[row, col]

            sns.boxplot(data=df, x=config, y=metric, hue="tracker", palette="Set2", ax=ax)
            sns.stripplot(data=df, x=config, y=metric, hue="tracker", dodge=True, jitter=True, palette='Set2', alpha=0.4, ax=ax)
        
            ax.grid(True)
            ax.set_ylabel(metric_labels[metric])
            ax.set_xlabel(f"Hyperparameter Configurations {order}")
            ax.set_xticks(range(len(ax.get_xticks())))
            ax.set_xticklabels(df[config].unique().tolist(), rotation=45, ha="right")
            if metric != 'num_switches':
                ax.set_ylim((-0.1, 1.1)) 
           
            # Calculate medians per group and annotate
            groups = df.groupby([config, "tracker"])

            n_hue = df["tracker"].nunique()
            xticks = ax.get_xticks()
            total_width = 0.8  # boxplot default

            for i, (key, group) in enumerate(groups):
                median_val = group[metric].median()

                x_index = i // n_hue
                hue_index = i % n_hue

                if x_index < len(xticks):
                    base_x = xticks[x_index]
                    box_width = total_width / n_hue
                    x = base_x - total_width/2 + box_width/2 + hue_index * box_width

                    y = median_val
                    if metric != "num_switches":
                        ax.text(x, y, f"{median_val:.3f}", ha='center', va='bottom',
                                fontsize=10, color='black', fontweight='bold')
                    else:
                        ax.text(x, y, f"{median_val}", ha='center', va='bottom',
                                fontsize=10, color='black', fontweight='bold')

            # Legends: keep only first subplot
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), [tracker_labels[l] for l in by_label.keys()], title="Tracker",)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            
        # Save plot
        fig.suptitle(f"{dataset} Dataset Train Metrics per Hyper-parameter Configuration ", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        plt.savefig(f"plots/{dataset}_combined_boxplot_{order}.pdf")
        print(f"Saved 'plots/{dataset}_combined_boxplot_{order}.pdf'")
        
        print("")

def load_grid_search_tracking_metrics():
    
    runs_dir = Path(__file__).resolve().parents[3] / 'runs' / 'tracking'

    experiment_folders = sorted([p for p in runs_dir.glob('*GridSearch') if p.is_dir()])

    grid_search_tracking_metrics_df : pd.DataFrame
    for i, experiment_folder in enumerate(experiment_folders):

        # Get Dataset & Tracker Names
        name = experiment_folder.name
        for dataset in DATASETS:
            if name.startswith(dataset):
                for tracker in TRACKERS:
                    expected_name = f"{dataset}{tracker}GridSearch"
                    if name == expected_name:
                        dataset_name = dataset 
                        tracker_name = tracker
                        break
                break
        else:
            print(f"Unrecognized dataset/tracker in: {name}")

        # Concatenate DataFrames & Include Dataset & Tracker Info
        exp_df = pd.read_csv(experiment_folder / 'experiment_tracking_metrics.csv')
        exp_df.insert(0, 'dataset', dataset_name)
        exp_df.insert(1, 'tracker', tracker_name)
        
        if i == 0:
            grid_search_tracking_metrics_df = exp_df 
        else:
            grid_search_tracking_metrics_df = pd.concat([grid_search_tracking_metrics_df, exp_df], ignore_index=True)
        
        print(f"Loaded '{name}/experiment_tracking_metrics.csv';")

    grid_search_tracking_metrics_df.to_csv(GRID_SEARCH_METRICS_PATH, index=False)

    print(f"\nSaved  {GRID_SEARCH_METRICS_PATH}.\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Hyperparameters GridSearch.")
    
    # Optional
    parser.add_argument('--metrics', nargs='+', default=['mota', 'motp', 'idf1', 'num_switches'], choices=['mota', 'motp', 'idf1', 'num_switches'], help="List of metrics to evaluate ['mota' 'motp' 'idf1' 'num_switches']"
    )
    parser.add_argument('--load', action='store_true', help="Set to True to load GridSearchTrackingMetrics.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():

    # === 1. Parse Arguments === #
    args    = parse_arguments()
    load    = args.load
    metrics = args.metrics

    # === 2. Load GridSearch Tracking Metrics === #
    if load:
        print("FILES:")
        print("------")
        load_grid_search_tracking_metrics()

    # === 3. Evaluate Hyperparameters By Dataset & Metrics === #
    print(f"PLOTS:")
    print(f"------")
    evaluate_hyperparameters(dataset="OpenWater", metrics=metrics)
    evaluate_hyperparameters(dataset="Coastline", metrics=metrics)
    evaluate_hyperparameters(dataset="Merged",    metrics=metrics)

if __name__ == "__main__":

    plt.rcParams.update({
        "axes.labelsize"  : 18, # axis labels
        "axes.titlesize"  : 18, # subplot titles
        "xtick.labelsize" : 10, # x tick labels
        "ytick.labelsize" : 16, # y tick labels
        "legend.fontsize" : 12, # legend text
        "figure.titlesize": 20  # figure title
    })

    main()