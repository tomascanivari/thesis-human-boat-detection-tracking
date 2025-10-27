import os
import argparse
import warnings 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from scipy.stats import wilcoxon, rankdata

DETECTION_METRICS_PATH = 'detection_metrics.csv'

DATASETS = ['Coastline', 'OpenWater', 'Merged', 'CoastlineV2Big']

MODELS = ['Coastline', 'OpenWater', 'Classification', 'Merged', 'CoastlineV2', 'CoastlineV2Big', 'ClassificationV2', 'ClassificationV2Big']

plt.rcParams.update({
    "axes.labelsize"  : 18, # axis labels
    "axes.titlesize"  : 18, # subplot titles
    "xtick.labelsize" : 16, # x tick labels
    "ytick.labelsize" : 16, # y tick labels
    "legend.fontsize" : 12, # legend text
    "figure.titlesize": 20  # figure title
})

metric_labels = {'precision': r"$\text{Precision}_{\text{det}}$", 'recall': r"$\text{Recall}_{\text{det}}$", 'f1-score': r"$\text{F1-score}_{\text{det}}$", 'mAP@50': r"$\text{mAP}_{50}$", 'mAP@50_95': r"$\text{mAP}_{50:95}$"}

models_labels = {'Coastline': 'Coastline', 
                    'OpenWater': 'OpenWater', 
                    'Merged': 'Merged', 
                    'Classification': 'Classification-OD', 
                    'CoastlineV2': 'ClassificationV2-OD',
                    'CoastlineV2Big': 'ClassificationV2Big-OD', 
                    'ClassificationV2': 'ClassificationV2-OD', 
                    'ClassificationV2Big': 'ClassificationV2Big-OD'}

A4_factor = np.sqrt(2)



def datasets_individual_evaluation(det_metrics_df: pd.DataFrame, dataset_model_pairs: list[(str, str)], metrics: list[str]):
    """
    Evaluate all splits of corresponding Model and Dataset.
    """

    # Ensure output directory exists
    os.makedirs(f"plots/IndividualEvaluation", exist_ok=True)

    plt.rcParams.update({
        "axes.labelsize"  : 18, # axis labels
        "axes.titlesize"  : 18, # subplot titles
        "xtick.labelsize" : 16, # x tick labels
        "ytick.labelsize" : 16, # y tick labels
        "legend.fontsize" : 12, # legend text
        "figure.titlesize": 20  # figure title
    })


    for dataset, model in dataset_model_pairs:

        # Filter dataframe for the correct dataset
        pair_df = det_metrics_df[(det_metrics_df['dataset'] == dataset) & (det_metrics_df['model'] == model)]
        
        pair_df = pair_df[(pair_df['split'] == 'train') | (pair_df['split'] == 'val')]

        # Ensure output directory exists
        os.makedirs(f"plots/IndividualEvaluation", exist_ok=True)

        fig_width = min(8, 8.27)

        fig_height = A4_factor * fig_width

        #  Combine into 2x2 grid 
        fig = plt.figure(figsize=(fig_width*2, fig_height))
        
        gs = gridspec.GridSpec(2, 6, figure=fig)
        
        axes = [
            fig.add_subplot(gs[0, 1:3]), fig.add_subplot(gs[0, 3:5]),                               # Top row
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6]),  # Bottom row
        ]

        for i, metric in enumerate(metrics):
                        
            ax = axes[i]

         # Check how many values exist per group
            group_sizes = pair_df.groupby(['split', 'class'])[metric].count()

            if (group_sizes > 1).any():
                sns.boxplot(data=pair_df, x='split', y=metric, hue='class', palette='Set2', ax=ax)
                sns.stripplot(data=pair_df, x='split', y=metric, hue='class',
                            dodge=True, jitter=True, palette='Set2', alpha=0.4, ax=ax)
                
                # Calculate median per group and annotate
                groups = pair_df.groupby(['split', 'class'])
                splits = pair_df['split'].unique()
                classes = pair_df['class'].unique()
                n_hue = len(classes)
                xticks = ax.get_xticks()
                total_width = 0.8  # default box width

                for i, ((split, cls), group) in enumerate(groups):
                    median_val = group[metric].median()

                    x_index = list(splits).index(split)
                    hue_index = list(classes).index(cls)

                    if x_index < len(xticks):
                        base_x = xticks[x_index]
                        box_width = total_width / n_hue
                        x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                        y = median_val
                        ax.text(x, y - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                                f"{median_val:.3f}", ha='center', va='bottom',
                                fontsize=10, color='black', fontweight='bold')

            else:
                # --- Case: single value -> dodged scatter points ---
                classes = pair_df['class'].unique()
                splits = pair_df['split'].unique()
                n_classes = len(classes)
                width = 0.8  # spread of the dodge

                # consistent colors from seaborn
                palette = dict(zip(classes, sns.color_palette("Set2", n_classes)))

                for split_idx, split in enumerate(splits):
                    for class_idx, cls in enumerate(classes):
                        subset = pair_df[(pair_df['split'] == split) & (pair_df['class'] == cls)]
                        if subset.empty:
                            continue

                        x_offset = split_idx + (class_idx - (n_classes - 1)/2) * (width / n_classes) 
                        y_val = subset[metric].values[0]

                        ax.scatter(
                            x_offset, y_val, s=80, color=palette[cls],
                            label=cls if split_idx == 0 else "", zorder=3
                        )

                        # Annotate value above point
                        ax.text(
                            x_offset, y_val + 0.05 * (pair_df[metric].max() - pair_df[metric].min() + 1e-6),
                            f"{y_val:.3f}", ha="center", va="bottom", fontweight="bold"
                        )
               
                ax.margins(x=0.2)
                ax.set_xticks(range(len(splits)))
                ax.set_xticklabels(splits)


            # ax.set_title(f"{metric_labels[metric]}")
            ax.set_ylabel(metric_labels[metric])
            ax.set_xlabel("Split")
            ax.grid(True)

            ax.set_ylim((-0.1, 1.1))

            # Deduplicate legend (only keep for first subplot)

            # handles, labels = ax.get_legend_handles_labels()
            # if i == 0:
            #     by_label = dict(zip(labels, handles))
            #     ax.legend(by_label.values(), by_label.keys(), title="Class")
            # else:
            #     ax.get_legend().remove()
            # Deduplicate legends (keep one per subplot)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), title="Class", loc="best")

            

        fig.suptitle(f"{dataset} Dataset - {model} Model - Train and Val Metrics by Class", fontweight="bold")
        plt.subplots_adjust(top=0.9)  # make room for the title
        plt.tight_layout()
        combined_path = f"plots/IndividualEvaluation/{dataset.lower()}_train_val_combined_boxplots.pdf"
        plt.savefig(combined_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {combined_path}\n")

def datasets_test_evaluation_by_class(det_metrics_df: pd.DataFrame, datasets: list[str], models: list[str], classes: list[str], metrics: list[str], folder_name: str):

    # Filter dataframe only to 'test' split
    det_metrics_df = det_metrics_df[(det_metrics_df['split'] == 'test')]

    # Update models labels 
    det_metrics_df.loc[(det_metrics_df["dataset"] == "Coastline") & (det_metrics_df["model"] == "CoastlineV2"), "model"] = "ClassificationV2"
    det_metrics_df.loc[(det_metrics_df["dataset"] == "Coastline") & (det_metrics_df["model"] == "CoastlineV2Big"), "model"] = "ClassificationV2Big"

    # Filter dataframe to only have the datasets and models required
    det_metrics_df = det_metrics_df[(det_metrics_df['dataset'].isin(datasets)) & (det_metrics_df['model'].isin(models))]


    # Ensure output directory exists
    os.makedirs(f"plots/TestEvaluation/{folder_name}", exist_ok=True)

    A4_factor = np.sqrt(2)

    fig_width = min(8.27, 8.27)

    fig_height = A4_factor * fig_width

    plt.rcParams.update({
        "legend.fontsize" : 6, # legend text
    })


    n_rows = len(metrics)
    n_cols = len(classes)


    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width*1.5, fig_height), sharex="col"
    )

    for col, class_name in enumerate(classes):


        # If only 1 row/col, make axes always 2D array
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        class_df = det_metrics_df[det_metrics_df['class'] == class_name]

        for row, metric in enumerate(metrics):
            ax = axes[row, col]

            # Check how many values exist per group
            group_sizes = class_df.groupby(['dataset', 'model'])[metric].count()


            # --- Multiple values -> boxplot + stripplot ---
            sns.boxplot(
                data=class_df, x='dataset', y=metric, hue='model',
                palette='Set2', order=datasets, hue_order=models, ax=ax
            )
            sns.stripplot(
                data=class_df, x='dataset', y=metric, hue='model',
                order=datasets, hue_order=models, ax=ax,
                dodge=True, jitter=True, palette='Set2', alpha=0.4
            )

            # Calculate median per group and annotate
            groups = class_df.groupby(['dataset', 'model'])
            n_hue = len(models)
            xticks = ax.get_xticks()
            total_width = 0.8

            # Store medians per dataset for connecting lines
            dataset_medians = {}

            for (dataset_, model_), gdf in groups:
                if dataset_ not in datasets or model_ not in models:
                    continue
                median_val = gdf[metric].median()
                x_index = list(datasets).index(dataset_)
                hue_index = list(models).index(model_)

                if x_index < len(xticks):
                    base_x = xticks[x_index]
                    box_width = total_width / n_hue
                    x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width
                    y = median_val
                    ax.text(
                        x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{median_val:.3f}", ha='center', va='bottom',
                        color='black', fontweight='bold'
                    )

                    # Save for connecting line
                    dataset_medians.setdefault(dataset_, []).append((x, y))

            # Draw dashed lines and annotate differences for each dataset
            for dataset_, values in dataset_medians.items():
                if len(values) >= 2:
                    # sort by x (left to right, i.e. model order in the plot)
                    values.sort(key=lambda tup: tup[0])

                   # loop through adjacent pairs
                    for (x1, y1), (x2, y2) in zip(values[:-1], values[1:]):
                        # Draw dashed line
                        ax.plot([x1, x2], [y1, y2], linestyle="--", color="gray", zorder=2)

                        # Compute differences
                        diff = y2 - y1
                        perc = (diff / y1 * 100) if y1 != 0 else float("inf")

                        # Midpoint in x
                        xm = (x1 + x2) / 2
                        # Midpoint in y shifted downward
                        ym = (y1 + y2) / 2 - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])

                        # Place annotation below the line
                        ax.text(
                            xm, ym,
                            f"{perc:+.1f}%",
                            ha="center", va="top", fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6)
                        )

            if row == 0:
                ax.set_title(f"{class_name.capitalize()} Class")

            # Y-axis: show only on first column
            if col == 0:
                ax.set_ylabel(metric_labels[metric])
                ax.yaxis.set_tick_params(labelleft=True)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.yaxis.set_tick_params(labelleft=False)

            # X-label only on bottom row
            if row == n_rows - 1:
                ax.set_xlabel("")
                ax.set_xticks(range(len(datasets)))
                ax.set_xticklabels(datasets)
                if col == 1:
                    ax.set_xticks(range(len(datasets)))
                    ax.set_xticklabels(datasets)
                    ax.set_xlabel("Dataset")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.grid(True)
            ax.set_ylim((-0.1, 1.1))

            # Legends: keep only first subplot
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), [models_labels[k] for k in by_label.keys()], title="Model")
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

            

    os.makedirs(f"plots/TestEvaluation/{folder_name}", exist_ok=True)
    fig.suptitle(f"Test Metrics by Dataset, Model and Class", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = f"plots/TestEvaluation/{folder_name}/combined_boxplot_all_classes.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
        
    print("")

def wilcoxon_test(det_metrics_df: pd.DataFrame):
    
    wilcoxon_df = pd.DataFrame(columns=['Dataset', 'Model A', 'Model B', 'Statistic', 'p-value', 'Values A', 'Values B'])

    # Filter dataframe to only have the class_name class, 'test' split and 'map50' metric
    df = det_metrics_df[(det_metrics_df['class'] == 'all') & (det_metrics_df['split'] == 'test')]
    
    df_openwater = df[(df['model'] == 'OpenWater')]

    df_merged = df[(df['model'] == 'Merged')]
    
    df_coastline        = df[(df['model'] == 'Coastline')]
    df_coastline_v2     = df[(df['model'] == 'CoastlineV2')]
    df_coastline_v2_big = df[(df['model'] == 'CoastlineV2Big')]
    
    df_classification        = df[(df['model'] == 'Classification')]
    df_classification_v2     = df[(df['model'] == 'ClassificationV2')]
    df_classification_v2_big = df[(df['model'] == 'ClassificationV2Big')]


    for dataset in ['Coastline', 'OpenWater', 'Merged']:

        # print(f"\033[1m{dataset.upper()} DATASET\033[0m")

        map_cst = df_coastline[(df_coastline['dataset'] == dataset)]['mAP@50'].values
        map_opw = df_openwater[(df_openwater['dataset'] == dataset)]['mAP@50'].values
        map_cls = df_classification[(df_classification['dataset'] == dataset)]['mAP@50'].values
        map_mrg = df_merged[(df_merged['dataset'] == dataset)]['mAP@50'].values
        
        pairs = {
            'names' : [('Coastline', 'OpenWater'), ('Coastline', 'Merged'), ('Openwater', 'Merged'), ('Classification', 'Merged')],
            'values': [(map_cst, map_opw), (map_cst, map_mrg), (map_opw, map_mrg), (map_cls, map_mrg)]
        }

        if dataset == 'Coastline':
            map_cst_v2     = df_coastline_v2[(df_coastline_v2['dataset'] == dataset)]['mAP@50'].values
            map_cst_v2_big = df_coastline_v2_big[(df_coastline_v2_big['dataset'] == dataset)]['mAP@50'].values

            pairs['names'].extend([('Coastline', 'CoastlineV2'), ('Coastline', 'CoastlineV2Big'), ('CoastlineV2', 'CoastlineV2Big')])
            pairs['values'].extend([(map_cst, map_cst_v2), (map_cst, map_cst_v2_big), (map_cst_v2, map_cst_v2_big)])

        elif dataset == 'Merged':
            map_cls_v2     = df_classification_v2[(df_classification_v2['dataset'] == dataset)]['mAP@50'].values
            map_cls_v2_big = df_classification_v2_big[(df_classification_v2_big['dataset'] == dataset)]['mAP@50'].values

            pairs['names'].extend([('Classification', 'ClassificationV2'), ('Classification', 'ClassificationV2Big'), ('ClassificationV2', 'ClassificationV2Big')])
            pairs['values'].extend([(map_cls, map_cls_v2), (map_cls, map_cls_v2_big), (map_cls_v2, map_cls_v2_big)])

        for names, values in zip(pairs['names'], pairs['values']):
            
            name_A, name_B = names
            value_A, value_B = values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(value_A, value_B, method='exact')
            

            # print(f"{name_A.upper():>14} PIPELINE: {value_A}")
            # print(f"{name_B.upper():>14} PIPELINE: {value_B}")    
            # print(f"WILCOXON TEST STATISTIC: {stat}, p-value: {p}\n\n")

            new_row = {'Dataset': dataset, 'Model A': name_A, 'Model B': name_B, 'Statistic': stat, 'p-value': p, 'Values A': value_A, 'Values B': value_B}
            if wilcoxon_df.empty:
                wilcoxon_df = pd.DataFrame([new_row])
            else:
                wilcoxon_df = pd.concat([wilcoxon_df, pd.DataFrame([new_row])], ignore_index=True)

    wilcoxon_df.to_csv('wilcoxon_test_mAP@50.csv', index=False)

    print(f"Saved 'wilcoxon_test_mAP@50.csv'")

def wilcoxon_test_with_plot(det_metrics_df: pd.DataFrame, datasets: list[str], models: list[str], split: str, class_name: str, metric: str):

    plt.rcParams.update({
        "axes.labelsize"  : 16, # axis labels
        "axes.titlesize"  : 16, # subplot titles
        "xtick.labelsize" : 14, # x tick labels
        "ytick.labelsize" : 14, # y tick labels
        "legend.fontsize" : 10, # legend text
        "figure.titlesize": 18,  # figure title
    })

    # Ensure output directory exists
    os.makedirs(f"plots/Wilcoxon/", exist_ok=True)

    # Define Wilcoxon DataFrame
    wilcoxon_df = pd.DataFrame(columns=['Dataset', 'Model A', 'Model B', 'Statistic', 'p-value', 'Values A', 'Values B'])

    # Filter dataframe to only have the class_name class, 'split' split
    df = det_metrics_df[(det_metrics_df['class'] == class_name) & (det_metrics_df['split'] == split)]
    
    # Update models labels 
    df.loc[(df["dataset"] == "Coastline") & (df["model"] == "CoastlineV2Big"), "model"] = "ClassificationV2Big"

    metric_values_per_dataset_model = {}

    for dataset in datasets:
        if dataset not in metric_values_per_dataset_model.keys():
            metric_values_per_dataset_model[dataset] = {}
        for model in models:
            metric_values_per_dataset_model[dataset][model] = df[(df['model'] == model) & (df['dataset'] == dataset)][metric].values
   

    folds = [i for i in range(6)]

    A4_factor = np.sqrt(2)

    fig_width = min(7.8, 8.27)

    fig_height = A4_factor * fig_width

    # Prepare a single 3x2 grid
    fig, axs = plt.subplots(len(datasets), len(models), figsize=(fig_width*2, fig_height*1))

    for idx, dataset in enumerate(datasets):

        # Wilcoxon Setup
        model_a, model_b =  metric_values_per_dataset_model[dataset].keys()
        values_a, values_b = metric_values_per_dataset_model[dataset][model_a], metric_values_per_dataset_model[dataset][model_b]

        model_a, model_b = models_labels[model_a], models_labels[model_b]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = wilcoxon(values_a, values_b, method='exact')

        new_row = {
            'Dataset': dataset,
            'Model A': model_a,
            'Model B': model_b,
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
        ax_left.scatter(folds, A, color='blue', label=model_a, s=100)
        ax_left.scatter(folds, B, color='red', label=model_b, s=100)

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
    plt.savefig("plots/Wilcoxon/wilcoxonv2_plots.pdf")

    wilcoxon_df.to_csv('wilcoxon_test_with_plots_mAP@50.csv', index=False)

    print(f"Saved 'wilcoxon_test_with_plots_mAP@50.csv'.")
        
def load_detection_metrics():
    
    runs_dir = Path(__file__).resolve().parents[2] / 'runs' / 'detection'

    experiment_folders = sorted([p for p in runs_dir.glob('*Model') if p.is_dir()])

    detection_metrics_df : pd.DataFrame
    for i, experiment_folder in enumerate(experiment_folders):

        # Get Dataset & Tracker Names
        name = experiment_folder.name

        # Concatenate DataFrames & Include Dataset & Tracker Info
        exp_df = pd.read_csv(experiment_folder / 'all_detection_metrics.csv')
        
        if i == 0:
            detection_metrics_df = exp_df 
        else:
            detection_metrics_df = pd.concat([detection_metrics_df, exp_df], ignore_index=True)
        
        print(f"Loaded '{name}/all_detection_metrics.csv';")

    detection_metrics_df.to_csv(DETECTION_METRICS_PATH, index=False)

    print(f"\nSaved  {DETECTION_METRICS_PATH}.\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Hyperparameters GridSearch.")
    
    # Optional
    parser.add_argument(
        '--metrics', 
        nargs='+', 
        default=['precision', 'recall', 'f1-score', 'mAP@50', 'mAP@50_95'], 
        choices=['precision', 'recall', 'f1-score', 'mAP@50', 'mAP@50_95'], 
        help="List of metrics to evaluate ['precision', 'recall', 'f1-score', 'mAP@50', 'mAP@50_95']"
    )
    
    parser.add_argument('--load', action='store_true', help="Set to True to load DetectionMetrics.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

def main():

    # === 1. Parse Arguments === #
    args    = parse_arguments()
    load    = args.load
    metrics = args.metrics

    # === 2. Load Detection Metrics === #
    if load:
        print("FILES:")
        print("------")
        load_detection_metrics()

    # === 3. Evaluate Metrics === #
    print(f"PLOTS:")
    print(f"------")

    # Load CSV 
    det_metrics_df = pd.read_csv(DETECTION_METRICS_PATH)

    # Convert all relevant columns to numeric
    for metric in metrics:
        det_metrics_df[metric] = pd.to_numeric(det_metrics_df[metric], errors='coerce')

    # === 3.1 Evaluate All Splits Of Corresponding Model & Dataset By Class ===
    # datasets_individual_evaluation(
    #     det_metrics_df=det_metrics_df,
    #     dataset_model_pairs=[('Coastline', 'Coastline'), ('OpenWater', 'OpenWater'), ('Merged', 'Merged'), ('CoastlineV2', 'CoastlineV2'), ('CoastlineV2Big', 'CoastlineV2Big')],
    #     metrics=metrics
    # )

    # # === 3.2 Compare Test Split Of Datasets By Models === #
    # datasets_test_evaluation_by_class(
    #     det_metrics_df=det_metrics_df,
    #     datasets=['Coastline', 'OpenWater', 'Merged'], 
    #     models=['Coastline', 'OpenWater', 'Classification', 'Merged'],
    #     classes=['all', 'swimmer', 'boat'],
    #     metrics=metrics,
    #     folder_name="CoastlineOpenWaterMergedDatasetsV1Models"
    # )

    # # # === 3.3 Compare Test Split Of V1, V2 & V2Big on Coastline and Merged By Class ===
    datasets_test_evaluation_by_class(
        det_metrics_df=det_metrics_df,
        datasets=['Coastline', 'Merged'], 
        models=['Classification', 'ClassificationV2', 'ClassificationV2Big'],
        classes=['all', 'swimmer', 'boat'],
        metrics=metrics,
        folder_name="CoastlineMergedDatasetsV2Models"
    )

    # # wilcoxon_test(det_metrics_df=det_metrics_df)

    # wilcoxon_test_with_plot(
    #     det_metrics_df=det_metrics_df,
    #     datasets=['Coastline', 'Merged'],
    #     models=['Classification', 'ClassificationV2Big'],
    #     split='test',
    #     class_name='all',
    #     metric='mAP@50')

if __name__ == "__main__":
    main()