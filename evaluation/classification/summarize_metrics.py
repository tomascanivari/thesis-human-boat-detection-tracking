import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

dataset = "ClassificationV2"

for dataset in ["Classification", "ClassificationV2"]:
    for split in ["train", "val", "test"]:
        # -----------------------------
        # Configuration
        # -----------------------------
        normalize = False
        fmt = ".4f" if normalize else "d"
        suffix = " (normalized)" if normalize else ""
        folds_dir = Path(f"{split}/{dataset}")
        num_folds = 6
        class_names = ["Coastline", "OpenWater"]

        # -----------------------------
        # Load confusion matrices and metrics
        # -----------------------------
        cms = []
        precisions, recalls, f1s, accuracies = [], [], [], []

        for fold in range(num_folds):
            # Load CM
            cm_path = folds_dir / f"fold{fold}_cm.csv"
            cm_df = pd.read_csv(cm_path, index_col=0)
            cm_values = cm_df.values
            cms.append(cm_values)

            # Load metrics JSON
            metrics_path = folds_dir / f"fold{fold}_metrics.json"
            with open(metrics_path) as f:
                metrics = json.load(f)

            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1-score"])
            accuracies.append(metrics["precision"]["accuracy"])  # same in all dicts

        # -----------------------------
        # Plot confusion matrices per fold
        # -----------------------------

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for i, cm in enumerate(cms):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
            ax = axes[i // 3, i % 3]
            sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 12})
            ax.set_xlabel("Predicted", fontsize=14)
            ax.set_ylabel("True", fontsize=14)
            ax.set_title(f"Fold {i}")
            ax.set_xticklabels(class_names, fontsize=12, rotation=0)
            ax.set_yticklabels(class_names, fontsize=12, rotation=0)

        fig.suptitle(f"{dataset} {split.upper()} Confusion Matrices{suffix}", fontsize=18, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        plt.savefig(f"plots/{dataset}Dataset/{split}_confusion_matrices_{suffix.strip(' ()')}.pdf", bbox_inches="tight")
        print(f"Saved plots/{dataset}Dataset/{split}_confusion_matrices_{suffix.strip(' ()')}.pdf")

        # -----------------------------
        # Plot overall confusion matrix
        # -----------------------------
        overall_cm = np.sum(cms, axis=0)

        overall_cm = overall_cm.astype(float) / overall_cm.sum(axis=1, keepdims=True) if normalize else overall_cm

        plt.figure(figsize=(6,5))
        sns.heatmap(overall_cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dataset} {split.upper()} Overall Confusion Matrix{suffix}", fontsize=12, weight="bold")
        plt.savefig(f"plots/{dataset}Dataset/{split}_overall_confusion_matrix_{suffix.strip(' ()')}.pdf", bbox_inches="tight")
        print(f"Saved plots/{dataset}Dataset/{split}_overall_confusion_matrix_{suffix.strip(' ()')}.pdf")
        
        TN, FP, FN, TP = overall_cm.ravel()

        # class 1 as positive
        prec1 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec1  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_1  = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

        # class 0 as positive
        prec0 = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        rec0  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_0  = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

        # macro-averaged
        prec_macro = (prec0 + prec1) / 2
        rec_macro  = (rec0 + rec1) / 2
        f1_macro   = (f1_0 + f1_1) / 2

        # overall accuracy
        accuracy = (TP + TN) / overall_cm.sum()

        print(f"Overall Metrics (micro-averaged)")
        print(f"Class 0 - Precision: {prec0:.6f}, Recall: {rec0:.6f}, F1: {f1_0:.6f}")
        print(f"Class 1 - Precision: {prec1:.6f}, Recall: {rec1:.6f}, F1: {f1_1:.6f}")
        print(f"Macro Avg - Precision: {prec_macro:.6f}, Recall: {rec_macro:.6f}, F1: {f1_macro:.6f}")
        print(f"Accuracy: {accuracy:.6f}")

        # Build dataframe
        df = pd.DataFrame({
            "Class": ["Coastline", "OpenWater", "Macro Avg", "Accuracy"],
            "Precision": [f"{prec0:.6f}", f"{prec1:.6f}", f"{prec_macro:.6f}", ""],
            "Recall": [f"{rec0:.6f}", f"{rec1:.6f}", f"{rec_macro:.6f}", ""],
            "F1": [f"{f1_0:.6f}", f"{f1_1:.6f}", f"{f1_macro:.6f}", ""],
            "Accuracy": ["", "", "", f"{accuracy:.6f}"]
        })

        # Save to CSV
        df.to_csv(f"{dataset}Dataset_{split}_overall_metrics.csv", index=False)

        # -----------------------------
        # Statistical analysis of metrics
        # -----------------------------
        # Convert to DataFrames
        precisions_df = pd.DataFrame(precisions)
        recalls_df = pd.DataFrame(recalls)
        f1s_df = pd.DataFrame(f1s)

        # Helper for mean ± std (and optionally CI95)
        def summarize(df):
            mean = df.mean()
            std = df.std()
            ci95 = 1.96 * std / np.sqrt(len(df))
            return mean.map('{:.6f}'.format) + " ± " + std.map('{:.6f}'.format) + " (±" + ci95.map('{:.6f}'.format) + ")"

        summary = pd.DataFrame({
            "precision": summarize(precisions_df),
            "recall": summarize(recalls_df),
            "f1-score": summarize(f1s_df)
        }).transpose()

        # Add overall accuracy row
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        acc_ci95 = 1.96 * acc_std / np.sqrt(len(accuracies))
        summary.loc["accuracy"] = [f"{acc_mean:.6f} ± {acc_std:.6f} (±{acc_ci95:.6f})"] + [""] * (summary.shape[1]-1)

        # Reset index so row labels become a proper column
        summary = summary.reset_index().rename(columns={"index": "metric"})

        print(summary)

        # save summary to CSV
        summary.to_csv(f"{dataset}Dataset_{split}_metrics_summary.csv", index=False)

        # -----------------------------
        # Boxplots of metrics across folds (by class)
        # -----------------------------
        # Keep only the real classes (avoid accuracy/macro avg/weighted avg columns)
        precisions_df_cls = precisions_df[class_names].copy()
        recalls_df_cls = recalls_df[class_names].copy()
        f1s_df_cls = f1s_df[class_names].copy()

        # Melt each metric DataFrame to long form (only classes)
        precision_melted = precisions_df_cls.melt(var_name="class", value_name="value")
        precision_melted["metric"] = "precision"

        recall_melted = recalls_df_cls.melt(var_name="class", value_name="value")
        recall_melted["metric"] = "recall"

        f1_melted = f1s_df_cls.melt(var_name="class", value_name="value")
        f1_melted["metric"] = "f1-score"

        # Combine into single long-form DataFrame
        metrics_long = pd.concat([precision_melted, recall_melted, f1_melted], ignore_index=True)

        # Plot grouped boxplots
        plt.figure(figsize=(10,6))   

        sns.boxplot(data=metrics_long, x='metric', y="value", hue='class', palette='Set2')
        sns.stripplot(data=metrics_long, x='metric', y="value", hue='class',
                    dodge=True, jitter=True, palette='Set2', alpha=0.4)

        # Fix legend (avoid duplicate from box + strip)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:len(class_names)+1], labels[:len(class_names)+1], title="Class")

        # Get current axis
        ax = plt.gca()

        # Calculate median per group and annotate
        groups = metrics_long.groupby(['metric', 'class'])
        metrics_ = metrics_long['metric'].unique()
        classes_ = metrics_long['class'].unique()
        n_hue = len(classes_)
        xticks = ax.get_xticks()
        total_width = 0.8  # default box width

        for i, ((metric_, cls), group) in enumerate(groups):
            median_val = group["value"].median()

            x_index = list(metrics_).index(metric_)
            hue_index = list(classes_).index(cls)

            if x_index < len(xticks):
                base_x = xticks[x_index]
                box_width = total_width / n_hue
                x = base_x - total_width / 2 + box_width / 2 + hue_index * box_width

                y = median_val
                ax.text(x, y + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # slight offset above bar
                        f"{median_val:.3f}", ha='center', va='bottom',
                        fontsize=8, color='black', fontweight='bold')

        plt.ylabel("Score")
        plt.title(f"{dataset} {split.upper()} Metrics distribution across folds ({split})", fontsize=14, weight="bold")
        plt.savefig(f"plots/{dataset}Dataset/{split}_metrics_boxplot_by_class.pdf", bbox_inches="tight")
        print(f"Saved plots/{dataset}Dataset/{split}_metrics_boxplot_by_class.pdf")

        #
        #
        #
        #



        print("===============================================================================\n")