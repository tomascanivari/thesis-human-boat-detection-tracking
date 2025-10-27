import os
import re
import warnings 
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import wilcoxon, rankdata

TEST_METRICS_PATH = 'tracking_test_metrics.csv'

DATASETS = ['Coastline', 'OpenWater', 'Merged']
TRACKERS = ['BotSort', 'ByteTrack']
MODELS   = ['Classification', 'ClassificationV2', 'ClassificationV2Big']


models_labels = {'Classification': 'Classification-OD', 
                 'ClassificationV2': 'ClassificationV2-OD', 
                 'ClassificationV2Big': 'ClassificationV2Big-OD'}

tracker_labels = {'BotSort': 'BoT-SORT', 
                  'ByteTrack': 'ByteTrack'}

metric_labels = {'mota'    : 'MOTA',
                 'motp'    : 'MOTP',
                 'idf1'    : r"$\text{F1-score}_{\text{trk}}$",
                 'IDS/1000': 'IDS/1000'}

def dataset_evaluation_by_metrics(df, dataset, model, metrics, class_names):

    # Ensure output directory exists
    os.makedirs(f"plots/DatasetEvaluation", exist_ok=True)
        
    # Filter dataframe for the correct dataset
    df = df[(df['dataset'] == dataset) & (df['model'] == model)]

    for class_name in class_names:

        melted = df[df['class'] == class_name]

        # Prepare botsort_values long-form dataframe for plotting
        melted = melted.melt(
            id_vars=["tracker", "fold"],
            value_vars=metrics,
            var_name="metric",
            value_name="value"
        )       

        # Scale only the rows where the metric is "num_switches" by 1000 and rename the metric
        melted.loc[melted["metric"] == "num_switches", "value"] /= 1000
        
        melted["metric"] = melted["metric"].replace({"num_switches": "num_switches/1000"})

        # Plot using seaborn boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=melted, x="metric", y="value", hue="tracker", palette='Set2')
        sns.stripplot(data=melted, x="metric", y="value", hue="tracker",
                            dodge=True, jitter=False, palette='Set2', alpha=0.4)

        plt.title(f"Comparison of Tracking Metrics by Tracker on {dataset} Dataset {model} Model '{class_name}' Class ")
        plt.ylabel("Metric Value")
        plt.xlabel("Metric")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.ylim((-0.4, 1))
        plt.tight_layout()
        
        # Deduplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Tracker", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Get current axis
        ax = plt.gca()

        # Calculate median per group and annotate
        groups = melted.groupby(['metric', 'tracker'])
        unique_metric = melted['metric'].unique()
        unique_tracker = melted['tracker'].unique()
        n_hue = len(unique_tracker)
        xticks = ax.get_xticks()
        total_width = 0.8  # default box width

        for i, ((metric, tracker), group) in enumerate(groups):
            median_val = group['value'].median()

            x_index = list(unique_metric).index(metric)
            hue_index = list(unique_tracker).index(tracker)

            if x_index < len(xticks):
                base_x = xticks[x_index]
                box_width = total_width / n_hue
                x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                y = median_val
                ax.text(x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                        f"{median_val:.3f}", ha='center', va='bottom',
                        fontsize=8, color='black', fontweight='bold')

        # Save plot
        plt.savefig(f"plots/DatasetEvaluation/{dataset}_{model}_{class_name}.png", dpi=300)
        plt.close()
        print(f"Saved plots/DatasetEvaluation/{dataset}_{model}_{class_name}.png")

def metric_evaluation_by_datasets(df, metric, model, class_names):

    # Ensure output directory exists
    os.makedirs(f"plots/MetricEvaluation", exist_ok=True)

    # Filter dataframe for the correct dataset
    df = df[(df['model'] == model)]

    for class_name in class_names:

        class_df = df[df['class'] == class_name]

        # Plot using seaborn boxplot   
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=class_df, x="dataset", y=metric, hue="tracker", palette='Set2', order=DATASETS)
        sns.stripplot(data=class_df, x="dataset", y=metric, hue="tracker", order=DATASETS,
                            dodge=True, jitter=False, palette='Set2', alpha=0.4)

        plt.title(f"Comparison of Tracker by Dataset '{metric}' Metric '{model}' Model")
        plt.ylabel(metric)
        plt.xlabel("Dataset")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Deduplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Tracker", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Get current axis
        ax = plt.gca()

        # Calculate median per group and annotate
        groups = class_df.groupby(['dataset', 'tracker'])
        unique_dataset = class_df['dataset'].unique()
        unique_tracker = class_df['tracker'].unique()
        n_hue = len(unique_tracker)
        xticks = ax.get_xticks()
        total_width = 0.8  # default box width

        for i, ((dataset, tracker), group) in enumerate(groups):
            median_val = group[metric].median()

            x_index = DATASETS.index(dataset)
            hue_index = list(unique_tracker).index(tracker)

            if x_index < len(xticks):
                base_x = xticks[x_index]
                box_width = total_width / n_hue
                x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                y = median_val
                ax.text(x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                        f"{median_val:.3f}", ha='center', va='bottom',
                        fontsize=8, color='black', fontweight='bold')

        # Save plot
        plt.savefig(f"plots/MetricEvaluation/{metric}_{model}_{class_name}.png", dpi=300)
        plt.close()
        print(f"Saved plots/MetricEvaluation/{metric}_{model}_{class_name}.png")

def compare_metrics_evaluation(df, metrics, datasets, class_name):

    # Ensure output directory exists
    os.makedirs(f"plots/CompareMetricEvaluation", exist_ok=True)

    plt.rcParams.update({
        "axes.labelsize"  : 16, # axis labels
        "axes.titlesize"  : 18, # subplot titles
        "xtick.labelsize" : 14, # x tick labels
        "ytick.labelsize" : 14, # y tick labels
        "legend.fontsize" : 12, # legend text
        "figure.titlesize": 18  # figure title
    })

    tracker_labels = {'BotSort': 'BoT-SORT', 'ByteTrack': 'ByteTrack'}
    metric_labels = {'mota': 'MOTA',
                     'motp': 'MOTP',
                     'idf1': r"$\text{F1-score}_{\text{trk}}$",
                     'num_switches': 'IDS'}

    n_rows = len(metrics)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7*n_cols, 4*n_rows), sharex="col"
    )

    # If only 1 row/col, make axes always 2D array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):

            # Filter dataframe for the correct dataset
            dataset_df = df[(df['dataset'] == dataset) & (df['class'] == class_name)]

            ax = axes[row, col]

            # Plot using seaborn boxplot   
            sns.boxplot(data=dataset_df, x="model", y=metric, hue="tracker", palette='Set2', order=MODELS, ax=ax)
            sns.stripplot(data=dataset_df, x="model", y=metric, hue="tracker", order=MODELS, ax=ax, dodge=True, jitter=False, palette='Set2', alpha=0.4)

            ax.set_title(f"{dataset} Dataset")
            ax.set_ylabel(metric_labels[metric])
            ax.set_xlabel("Model")
            ax.grid(True)
            if metric == 'mota':
                ax.set_ylim((-0.4, 1.1)) 
            elif metric in ['motp', 'idf1']:
                ax.set_ylim((-0.1, 1.1))
            else:
                ax.set_ylim((-200, 1000)) 
            
            # Legends: keep only first subplot
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), [tracker_labels[l] for l in by_label.keys()], title="Tracker",)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

            if col == 0:
                ax.set_ylabel(metric_labels[metric])
                ax.yaxis.set_tick_params(labelleft=True)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.yaxis.set_tick_params(labelleft=False)

            if row > 0:
                ax.set_title("")


            # Calculate median per group and annotate
            groups = dataset_df.groupby(['model', 'tracker'])
            unique_model = dataset_df['model'].unique()
            unique_tracker = dataset_df['tracker'].unique()
            n_hue = len(unique_tracker)
            xticks = ax.get_xticks()
            total_width = 0.8  # default box width


            # Store medians for each tracker
            tracker_medians = {t: {} for t in unique_tracker}
            for i, ((model, tracker), group) in enumerate(groups):
                median_val = group[metric].median()

                x_index = MODELS.index(model)
                hue_index = list(unique_tracker).index(tracker)

                if x_index < len(xticks):
                    base_x = xticks[x_index]
                    box_width = total_width / n_hue
                    x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                    y = median_val
                    ax.text(x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                            f"{median_val:.3f}", ha='center', va='bottom',
                            fontsize=14, color='black', fontweight='bold')

                    # Save the coordinates for connecting later
                    tracker_medians[tracker][model] = (x, y)    

            # --- Connect medians per tracker across models ---
            # Build palette mapping once
            palette = dict(zip(unique_tracker, sns.color_palette("Set2", len(unique_tracker))))
            for tracker, model_points in tracker_medians.items():
                if len(model_points) > 1:
                    # Sort by model order
                    sorted_points = [(tracker_medians[tracker][m][0], tracker_medians[tracker][m][1])
                                     for m in MODELS if m in model_points]

                    for (x1, y1), (x2, y2) in zip(sorted_points[:-1], sorted_points[1:]):
                        # draw line
                        ax.plot([x1, x2], [y1, y2], linestyle="--", color="gray", alpha=0.7)

                        # compute difference
                        diff = y2 - y1
                        pct_diff = (diff / (y1 + 1e-6)) * 100

                        color = palette[tracker]  # get tracker color

                        # annotate in the middle, slightly below line
                        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(xm, ym - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                                f"{diff:.3f}\n({pct_diff:+.1f}%)",
                                ha="center", va="top", fontsize=12, color=color, weight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", edgecolor="none", facecolor="white", alpha=0.6))
                
    # Save plot
    fig.suptitle(f"Comparison of Classification-OD Model Versions Test Metrics by Dataset and Tracker - {class_name.capitalize()} Class", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"plots/CompareMetricEvaluation/all_metrics_{class_name}_class.pdf")
    plt.close()
    print(f"Saved plots/CompareMetricEvaluation/all_metrics_{class_name}_class.pdf")

def all_datasets_metrics_classification_model(df):

    plt.rcParams.update({
        "axes.labelsize"  : 16, # axis labels
        "axes.titlesize"  : 18, # subplot titles
        "xtick.labelsize" : 14, # x tick labels
        "ytick.labelsize" : 14, # y tick labels
        "legend.fontsize" : 12, # legend text
        "figure.titlesize": 20  # figure title
    })

    classes = ['all', 'swimmer', 'boat']

    # Ensure output directory exists
    os.makedirs(f"plots/AllDatasetsEvaluation", exist_ok=True)

    tracker_labels = {'BotSort': 'BoT-SORT', 'ByteTrack': 'ByteTrack'}

    # Metrics Settings
    metrics = ['mota', 'motp', 'idf1', 'num_switches']

    # Figure Settings
    n_rows = len(DATASETS)
    n_cols = len(classes)
    col = 0

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5*n_cols, 1.2*4*n_rows), sharex="col"
    )

    # If only 1 row/col, make axes always 2D array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, dataset in enumerate(DATASETS):
        

        # Filter dataframe for the correct dataset
        temp_df =  df[(df['dataset'] == dataset) & (df['model'] == 'Classification')]

        for col, class_name in enumerate(classes):
            ax = axes[row, col]

            melted = temp_df[temp_df['class'] == class_name]

            # Prepare botsort_values long-form dataframe for plotting
            melted = melted.melt(
                id_vars=["tracker", "fold"],
                value_vars=metrics,
                var_name="metric",
                value_name="value"
            )       

            # Scale only the rows where the metric is "num_switches" by 1000 and rename the metric
            melted.loc[melted["metric"] == "num_switches", "value"] /= 1000
            
            melted["metric"] = melted["metric"].replace({"num_switches": "IDS/1000"})

            # Plot using seaborn boxplot
            sns.boxplot(data=melted, x="metric", y="value", hue="tracker", palette='Set2', ax=ax)
            sns.stripplot(data=melted, x="metric", y="value", hue="tracker", ax=ax, dodge=True, jitter=False, palette='Set2', alpha=0.4)
            
            ax.grid(True)
            ax.set_title(f"{dataset} Dataset - {class_name.capitalize()} Class")
            ax.set_ylabel("Value")
            ax.set_xlabel("Metrics")
            ax.set_xticks(range(len(ax.get_xticks())))
            ax.set_xticklabels(metric_labels[l] for l in melted["metric"].unique().tolist())
            ax.set_ylim((-0.4, 1))

            # Calculate median per group and annotate
            groups = melted.groupby(['metric', 'tracker'])
            unique_metric = melted['metric'].unique()
            unique_tracker = melted['tracker'].unique()
            n_hue = len(unique_tracker)
            xticks = ax.get_xticks()
            total_width = 0.8  # default box width

            for i, ((metric, tracker), group) in enumerate(groups):
                median_val = group['value'].median()

                x_index = list(unique_metric).index(metric)
                hue_index = list(unique_tracker).index(tracker)

                if x_index < len(xticks):
                    base_x = xticks[x_index]
                    box_width = total_width / n_hue
                    x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                    y = median_val
                    ax.text(x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                            f"{median_val:.3f}", ha='center', va='bottom',
                            fontsize=12, color='black', fontweight='bold')
        
            # Legends: keep only first subplot
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), [tracker_labels[l] for l in by_label.keys()] , title="Tracker",)
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            
            # X-label only on bottom row
            if row == n_rows - 1:
                ax.set_xlabel("")
                ax.set_xticks(range(len(metrics)))
                ax.set_xticklabels([metric_labels[m] for m in unique_metric])
                if col == 1:
                    ax.set_xticks(range(len(metrics)))
                    ax.set_xticklabels([metric_labels[m] for m in unique_metric])
                    ax.set_xlabel("Dataset")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])



    # Save plot
    fig.suptitle(f"Classification-OD Model Test Metrics by Dataset, Tracker and Class", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"plots/AllDatasetsEvaluation/all_datasets_classification.pdf")

def wilcoxon_test_with_plot(trk_metrics_df: pd.DataFrame, datasets: list[str], model: str, class_name: str, metric: str):

    plt.rcParams.update({
        "axes.labelsize"  : 16, # axis labels
        "axes.titlesize"  : 16, # subplot titles
        "xtick.labelsize" : 14, # x tick labels
        "ytick.labelsize" : 14, # y tick labels
        "legend.fontsize" : 10, # legend text
        "figure.titlesize": 18,  # figure title
    })

    # Define Wilcoxon DataFrame
    wilcoxon_df = pd.DataFrame(columns=['Dataset', 'Model A', 'Model B', 'Statistic', 'p-value', 'Values A', 'Values B'])

    # Update models labels 
    trk_metrics_df.loc[(trk_metrics_df["dataset"] == "OpenWater") & (trk_metrics_df["model"] == "Classification"), "model"] = "ClassificationV2Big"

    # Filter dataframe to only have the class_name class
    df = trk_metrics_df[(trk_metrics_df['class'] == class_name) & (trk_metrics_df['model'] == model)]
    
    metric_values_per_dataset_tracker = {}

    for dataset in datasets:
        if dataset not in metric_values_per_dataset_tracker.keys():
            metric_values_per_dataset_tracker[dataset] = {}
        for tracker in TRACKERS:
            metric_values_per_dataset_tracker[dataset][tracker] = df[(df['tracker'] == tracker) & (df['dataset'] == dataset)][metric].values
   
    folds = [i for i in range(6)]

    A4_factor = np.sqrt(2)

    fig_width = min(7.8, 8.27)

    fig_height = A4_factor * fig_width

    # Prepare a single 3x2 grid
    fig, axs = plt.subplots(len(datasets), 2, figsize=(fig_width*2, fig_height*1))

    for idx, dataset in enumerate(datasets):

        # Wilcoxon Setup
        tracker_a, tracker_b =  metric_values_per_dataset_tracker[dataset].keys()
        values_a, values_b = metric_values_per_dataset_tracker[dataset][tracker_a], metric_values_per_dataset_tracker[dataset][tracker_b]

        tracker_a, tracker_b = tracker_labels[tracker_a], tracker_labels[tracker_b]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = wilcoxon(values_a, values_b, method='exact')

        new_row = {
            'Dataset': dataset,
            'Model A': tracker_a,
            'Model B': tracker_b,
            'Statistic': stat,
            'p-value': p,
            'Values A': values_a,
            'Values B': values_b
        }
        if wilcoxon_df.empty:
            wilcoxon_df = pd.DataFrame([new_row])
        else:
            wilcoxon_df = pd.concat([wilcoxon_df, pd.DataFrame([new_row])], ignore_index=True)

        # Differences and percent differences
        A, B = values_a, values_b
        diff = A - B
        percent_diff = (diff / B) * 100  # percentage relative to Model B

        # Compute ranks of absolute differences
        abs_diff = np.abs(diff)
        ranks = rankdata(abs_diff)
        signed_ranks = ranks * np.sign(diff)
        W_pos = signed_ranks[signed_ranks > 0].sum()
        W_neg = -signed_ranks[signed_ranks < 0].sum()

        # --- Left plot: Paired values with differences ---
        ax_left = axs[idx, 0]
        for i in range(len(A)):
            ax_left.plot([folds[i], folds[i]], [A[i], B[i]], 'k--', alpha=0.5)
        ax_left.scatter(folds, A, color='blue', label=tracker_a, s=100)
        ax_left.scatter(folds, B, color='red', label=tracker_b, s=100)

        # Annotate differences + percent
        for i in range(len(A)):
            # Place annotation slightly above the higher value of the two models
            y_pos = A[i] + 0.02 if A[i] >= B[i] else A[i] - 0.06
            # y_pos = max(A[i], B[i]) + 0.02  # adjust 0.01 depending on your y-axis scale
            ax_left.text(folds[i], y_pos, f"{diff[i]:.3f} ({percent_diff[i]:.1f}%)",
                        ha='center', va='bottom', fontweight="bold", fontsize=12)
        ax_left.set_xlim(folds[0] - 1, folds[-1] + 1)  
        ax_left.set_xticks(folds)  
        ax_left.set_xlabel('Fold')
        ax_left.set_ylabel(metric_labels[metric])
        ax_left.set_title(f'{dataset} - Paired Differences')
        ax_left.legend(loc="lower left")
        ax_left.grid(alpha=0.3)

        y_min = min(A.min(), B.min())
        y_max = max(A.max(), B.max())
        ax_left.set_ylim(y_min - 0.1, y_max + 0.1)

        # --- Right plot: Signed ranks with sums ---
        ax_right = axs[idx, 1]
        colors = ['green' if sr > 0 else 'orange' for sr in signed_ranks]
        ax_right.bar(folds, signed_ranks, color=colors)
        for i, sr in enumerate(signed_ranks):
            val = sr + 0.2 if np.sign(sr) > 0 else sr - 2 
            ax_right.text(folds[i], val, f"{sr:.1f}", ha='center', fontsize=12, fontweight="bold")

        # Draw sum lines
        ax_right.axhline(W_pos, color='green', linestyle='--', linewidth=2.5, label=f'Positive sum = {W_pos}')
        ax_right.axhline(-W_neg, color='orange', linestyle='--', linewidth=2.5, label=f'Negative sum = {W_neg}')
        ax_right.axhline(0, color='black', linewidth=0.8)
        ax_right.set_xticks(folds)
        ax_right.set_xlabel('Fold')
        ax_right.set_ylabel('Signed Rank')
        ax_right.set_title(f'{dataset} - Wilcoxon Signed Ranks ($W = {stat}, p = {p:.3f}$)')
        ax_right.legend(loc="lower left")
        ax_right.grid(alpha=0.3)
        ax_right.set_ylim(-W_neg - 3, W_pos + 3)

    fig.suptitle(f"Paired Differences and Wilcoxon Signed Ranks by Dataset - {class_name.capitalize()} Class", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("tracking_wilcoxon_plots.pdf")

    wilcoxon_df.to_csv('wilcoxon_test_with_plots_mota.csv', index=False)

    print(f"Saved 'wilcoxon_test_with_plots_mota.csv'.")

def load_tracking_test_metrics():
    
    def extract_info(filename):
        pattern = r'^(Coastline|OpenWater|Merged)(BotSort|ByteTrack)Test(?:([A-Za-z0-9]+))?$'
        match = re.match(pattern, filename)
        
        if match:
            dataset, tracker, version = match.groups()
            return dataset, tracker, version or None
        else:
            raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    runs_dir = Path(__file__).resolve().parents[3] / 'runs' / 'tracking'

    experiment_folders = sorted([p for p in runs_dir.glob('*Test*') if p.is_dir()])

    test_tracking_metrics_df : pd.DataFrame
    for i, experiment_folder in enumerate(experiment_folders):

        # Get Dataset & Tracker Names
        name = experiment_folder.name
        dataset, tracker, version = extract_info(name)

        model = 'Classification' + version if version is not None else 'Classification'
        
        # Concatenate DataFrames & Include Dataset & Tracker Info
        try:
            exp_df = pd.read_csv(experiment_folder / 'experiment_tracking_metrics.csv')
            exp_df.insert(0, 'dataset', dataset)
            exp_df.insert(1, 'tracker', tracker)
            exp_df.insert(2, 'model', model)

        except Exception as e:
            print(e)
            continue

        if i == 0:
            test_tracking_metrics_df = exp_df 
        else:
            test_tracking_metrics_df = pd.concat([test_tracking_metrics_df, exp_df], ignore_index=True)
        
        print(f"Loaded '{name}/experiment_tracking_metrics.csv';")

    test_tracking_metrics_df.to_csv(TEST_METRICS_PATH, index=False)

    print(f"\nSaved  {TEST_METRICS_PATH}.\n")

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

    # === 2. Load Test Tracking Metrics === #
    if load:
        print("FILES:")
        print("------")
        load_tracking_test_metrics()

    # Load CSV 
    tracking_metrics_df = pd.read_csv(TEST_METRICS_PATH)

    # Convert all relevant columns to numeric
    for metric in metrics:
        tracking_metrics_df[metric] = pd.to_numeric(tracking_metrics_df[metric], errors='coerce')

    # === 3. Evaluate Datasets & Metrics === #
    # print(f"PLOTS:")
    # print(f"------")
    # for dataset in DATASETS:
    #     models = ['Classification'] if dataset == 'OpenWater' else MODELS
    #     for model in models:
    #         dataset_evaluation_by_metrics(tracking_metrics_df, dataset, model, metrics, ['all'])
    #     print("")

    # for metric in metrics:
    #     metric_evaluation_by_datasets(tracking_metrics_df, metric, 'Classification', ['all'])
    
    # print("")



    # === 4. Thesis Plots === #
    all_datasets_metrics_classification_model(tracking_metrics_df)

    compare_metrics_evaluation(tracking_metrics_df, metrics, ['Coastline', 'Merged'], 'all')

    wilcoxon_test_with_plot(tracking_metrics_df, ['Coastline', 'OpenWater', 'Merged'], 'ClassificationV2Big', 'all', 'mota')

if __name__ == "__main__":
    main()