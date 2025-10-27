#!/bin/bash

train="true"
val="true"
test="true"
bytetrack="true"
botsort="false"
deepsort="false"
dataset="SeaDroneSee-MOT"

# CoastlineDrone Dataset
if [[ $dataset = "CoastlineDrone-MOT" ]]; then
    # ByteTrack and BotSORT using ULTRALYTICS
    for tracker in bytetrack botsort; do
        if [[ ( "$tracker" = "bytetrack" && "$bytetrack" = "true" ) || ( "$tracker" = "botsort" && "$botsort" = "true" ) ]]; then
            for VAR in seq00 seq01 seq02 seq03 seq04 seq05 seq06 seq07 seq08 seq09 seq10 seq11; do
                echo -e "\e[1;34m$(figlet $VAR $tracker | boxes -d stone)\e[0m"
                python3 track.py --seq_name $VAR --dataset $dataset --model models/yolo12s_CoastlineDrone.pt --tracker $tracker.yaml --save_video --save_csv
                python3 helper_scripts/evaluate.py --gt_file $VAR.csv --pred_file $tracker/$VAR.csv --results_file $tracker/$VAR --dataset $dataset
            done
        fi
    done

    # DeepSORT using DETECTION from ULTRALYTICS and ORIGINAL DEEPSORT IMPLEMENTATION
    if [[ $deepsort = "true" ]]; then
        for VAR in seq00 seq01 seq02 seq03 seq04 seq05 seq06 seq07 seq08 seq09 seq10 seq11; do
            echo -e "\e[1;34m$(figlet $VAR deepsort | boxes -d stone)\e[0m"
            python3 detect.py --model models/yolo12s_CoastlineDrone.pt --seq_name $VAR --dataset $dataset
            python3 deep_sort/deep_sort_app.py --sequence_dir datasets/$dataset/sequences/$VAR --detection_file datasets/$dataset/detections/$VAR.npy --min_confidence 0.3 --nn_budget 100 --output_file results/$dataset/deepsort/$VAR.csv --display False
            python3 helper_scripts/evaluate.py --gt_file $VAR.csv --pred_file deepsort/$VAR.csv --results_file deepsort/$VAR --dataset $dataset
        done
    fi
fi

# SeaDroneSee Dataset
if [[ $dataset = "SeaDroneSee-MOT" ]]; then
    # ByteTrack and BotSORT using ULTRALYTICS
    for tracker in bytetrack botsort; do
        if [[ ( "$tracker" = "bytetrack" && "$bytetrack" = "true" ) || ( "$tracker" = "botsort" && "$botsort" = "true" ) ]]; then
            if [[ $train = "true" ]]; then
                echo -e "\e[1;31m$(figlet TRAIN SEQS | boxes -d stone)\e[0m"
                    for VAR in seq0 seq1 seq2 seq4 seq5 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq13 seq14 seq15 seq16 seq17 seq18 seq19 seq20 seq21; do
                        echo -e "\e[1;34m$(figlet $VAR $tracker | boxes -d stone)\e[0m"
                        python3 track.py --seq_name train/$VAR --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --tracker $tracker.yaml --save_video --save_csv
                        python3 helper_scripts/evaluate.py --gt_file train/$VAR.csv --pred_file $tracker/train/$VAR.csv --results_file $tracker/train/$VAR --dataset $dataset
                    done
            fi

            if [[ $val = "true" ]]; then
                echo -e "\e[1;31m$(figlet VAL SEQS | boxes -d stone)\e[0m"
                    for VAR in seq0 seq1 seq2 seq4 seq5 seq6 seq9 seq10 seq11 seq12 seq13 seq15 seq16 seq17 seq18 seq19 seq21; do
                        echo -e "\e[1;34m$(figlet $VAR $tracker | boxes -d stone)\e[0m"
                        python3 track.py --seq_name val/$VAR --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --tracker $tracker.yaml --save_video --save_csv
                        python3 helper_scripts/evaluate.py --gt_file val/$VAR.csv --pred_file $tracker/val/$VAR.csv --results_file $tracker/val/$VAR --dataset $dataset
                    done
            fi

            if [[ $test = "true" ]]; then
                echo -e "\e[1;31m$(figlet TEST SEQS | boxes -d stone)\e[0m"
                for VAR in seq0 seq1 seq2 seq3 seq4 seq5 seq6 seq9 seq10 seq11 seq12 seq13 seq14 seq15 seq16 seq17 seq18 seq19 seq21; do
                    echo -e "\e[1;34m$(figlet $VAR $tracker | boxes -d stone)\e[0m"
                    python3 track.py --seq_name test/$VAR --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --tracker $tracker.yaml --save_video --save_csv
                done
            fi
        fi
    done

    # DeepSORT using DETECTION from ULTRALYTICS and ORIGINAL DEEPSORT IMPLEMENTATION
    if [[ $deepsort = "true" ]]; then
        if [[ $train = "true" ]]; then
            echo -e "\e[1;31m$(figlet TRAIN SEQS | boxes -d stone)\e[0m"
                for VAR in seq0 seq1 seq2 seq4 seq5 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq13 seq14 seq15 seq16 seq17 seq18 seq19 seq20 seq21; do
                    echo -e "\e[1;34m$(figlet $VAR deepsort | boxes -d stone)\e[0m"
                    python3 detect.py --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --seq_name train/$VAR
                    python3 deep_sort/deep_sort_app.py --sequence_dir datasets/SeaDroneSee-MOT/sequences/train/$VAR --detection_file datasets/SeaDroneSee-MOT/detections/train/$VAR.npy --min_confidence 0.3 --nn_budget 100 --output_file results/SeaDroneSee-MOT/deepsort/train/$VAR.csv --display False
                    python3 helper_scripts/evaluate.py --gt_file train/$VAR.csv --pred_file deepsort/train/$VAR.csv --results_file deepsort/train/$VAR --dataset $dataset
                done
        fi

        if [[ $val = "true" ]]; then
            echo -e "\e[1;31m$(figlet VAL SEQS | boxes -d stone)\e[0m"
                for VAR in seq0 seq1 seq2 seq4 seq5 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq13 seq14 seq15 seq16 seq17 seq18 seq19 seq20 seq21; do
                    echo -e "\e[1;34m$(figlet $VAR deepsort | boxes -d stone)\e[0m"
                    python3 detect.py --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --seq_name val/$VAR
                    python3 deep_sort/deep_sort_app.py --sequence_dir datasets/SeaDroneSee-MOT/sequences/val/$VAR --detection_file datasets/SeaDroneSee-MOT/detections/val/$VAR.npy --min_confidence 0.3 --nn_budget 100 --output_file results/SeaDroneSee-MOT/deepsort/val/$VAR.csv --display False
                    python3 helper_scripts/evaluate.py --gt_file val/$VAR.csv --pred_file deepsort/val/$VAR.csv --results_file deepsort/val/$VAR --dataset $dataset
                done
        fi

        if [[ $test = "true" ]]; then
            echo -e "\e[1;31m$(figlet TEST SEQS | boxes -d stone)\e[0m"
                for VAR in seq0 seq1 seq2 seq4 seq5 seq6 seq7 seq8 seq9 seq10 seq11 seq12 seq13 seq14 seq15 seq16 seq17 seq18 seq19 seq20 seq21; do
                    echo -e "\e[1;34m$(figlet $VAR deepsort | boxes -d stone)\e[0m"
                    python3 detect.py --dataset $dataset --model models/yolo12s_SeaDroneSee.pt --seq_name test/$VAR
                    python3 deep_sort/deep_sort_app.py --sequence_dir datasets/SeaDroneSee-MOT/sequences/test/$VAR --detection_file datasets/SeaDroneSee-MOT/detections/test/$VAR.npy --min_confidence 0.3 --nn_budget 100 --output_file results/SeaDroneSee-MOT/deepsort/test/$VAR.csv --display False
                done
        fi
    fi
fi