import re
import os

from pathlib import Path
from collections import defaultdict


def dataset_analysis(dataset: str):

    # === 1. Define Folders === #
    dataset_folder = Path(f"datasets/{dataset}Dataset")
    folds_folder   = dataset_folder / "folds"

    # === 2. Load Folds TXT Files === #
    txt_files_path = sorted(folds_folder.glob('*txt'))

    pattern = re.compile(r'(train|val|test)_fold(\d+)')

    image_paths = defaultdict(dict)
    label_paths = defaultdict(dict)

    # Obtain Images and Labels Paths 
    for txt_file_path in txt_files_path:
        filename = os.path.basename(txt_file_path)
        match = pattern.match(filename)
        if match:
            split = match.group(1)
            fold = int(match.group(2))
            with open(txt_file_path, 'r') as f:
                imgs = [line.strip() for line in f if line.strip()]
                lbls = [p.replace('/images', '/labels') for p in imgs]
                lbls = [p.replace('jpg', 'txt') for p in lbls]
                image_paths[split][fold] = imgs
                label_paths[split][fold] = lbls

    # === 3. Process the Labels to Count Frames & Instances === #
    instances = defaultdict(dict)
    swimmer   = defaultdict(dict)
    boat      = defaultdict(dict)

    for split in label_paths.keys():
        # Create 'split' Key
        if split not in instances.keys():
            instances[split] = {}
            swimmer[split]   = {}
            boat[split]      = {}
        for fold in label_paths[split].keys():
            # Create 'fold' Key
            if fold not in instances[split].keys():
                instances[split][fold] = 0
                swimmer[split][fold]   = 0
                boat[split][fold]      = 0
            for label_path in label_paths[split][fold]:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        label, xc, yc, w, h = map(float, line.strip().split())
                        label = int(label)
                        instances[split][fold] += 1 
                        if label == 1:
                            swimmer[split][fold] += 1
                        elif label == 2:
                            boat[split][fold] += 1
    
    # === 4. Obtain Totals === #
    total_frames_by_fold = {
        fold: sum(len(image_paths[split].get(fold, [])) for split in ['train', 'val', 'test'])
        for fold in set(f for s in image_paths.values() for f in s)
    }

    def compute_fold_total(dictionary):
        # Compute total per fold across splits
        fold_totals = defaultdict(int)
        for split in dictionary.values():
            for fold, count in split.items():
                fold_totals[fold] += count

        # Convert to regular dict if needed
        fold_totals = dict(fold_totals)
        return fold_totals

    total_instances_by_fold  = compute_fold_total(instances)
    total_swimmer_by_fold    = compute_fold_total(swimmer)
    total_boat_by_fold       = compute_fold_total(boat)

    # === 5. Display & Log Information === #
    print("")
    print(f"{dataset.upper()} DATASET ANALYSYS:")
    print("-" * len(f"{dataset.upper()} DATASET ANALYSYS:"))

    HORIZONTAL_LINE = f"++=======++" + ("=" * 65 + "++") * 3
    print(f"         {HORIZONTAL_LINE[9:]}")
    print(f"         || {'Train'.center(63)} || {'Validation'.center(63)} || {'Test'.center(63)} ||")
    print(HORIZONTAL_LINE)
    print(f"|| {'Fold':<5} ||" + f" {'Frames (%)':>15} {'Instances (%)':>15} {'Swimmer (%)':>15} {'Boat (%)':>15} ||" * 3)
    print(HORIZONTAL_LINE)

    for fold in sorted(total_frames_by_fold):
        total_frames = total_frames_by_fold[fold]
        total_instances = total_instances_by_fold[fold]
        total_swimmer = total_swimmer_by_fold[fold]
        total_boat = total_boat_by_fold[fold]

        def fmt(count, total):
            return f"{count:>6} ({(count / total * 100):4.1f}%)" if total > 0 else "0 ( 0.0%)"

        train_count = len(image_paths['train'].get(fold, []))
        val_count   = len(image_paths['val'].get(fold, []))
        test_count  = len(image_paths['test'].get(fold, []))

        msg = f"|| {fold:<5} || {fmt(train_count, total_frames):>15} {fmt(instances['train'][fold], total_instances):>15} {fmt(swimmer['train'][fold], instances['train'][fold]):>15} {fmt(boat['train'][fold], instances['train'][fold]):>15} ||" + f" {fmt(val_count, total_frames):>15} {fmt(instances['val'][fold], total_instances):>15} {fmt(swimmer['val'][fold], instances['val'][fold]):>15} {fmt(boat['val'][fold], instances['val'][fold]):>15} ||" + f" {fmt(test_count, total_frames):>15} {fmt(instances['test'][fold], total_instances):>15} {fmt(swimmer['test'][fold], instances['test'][fold]):>15} {fmt(boat['test'][fold], instances['test'][fold]):>15} ||"
        print(msg)

    print(HORIZONTAL_LINE)

    print("")
    print(str(f"         ++" + ("=" * 65 + "++")))
    print(str(f"         ||" + f" {'Frames (%)':>15} {'Instances (%)':>15} {'Swimmer (%)':>15} {'Boat (%)':>15} ||"))
    print(str(f"++=======++" + ("=" * 65 + "++")))
    print(str(f"|| TOTAL || {fmt(total_frames, total_frames):>15} {fmt(total_instances, total_instances):>15} {fmt(total_swimmer, total_instances):>15} {fmt(total_boat, total_instances):>15} ||"))
    print(str(f"++=======++" + ("=" * 65 + "++")))

    return image_paths, label_paths