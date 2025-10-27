#!/bin/bash

# === 1. Cross Validation === #

# === 1.1 Coastline Dataset === #
# python3 evaluate_tracking_folds.py --dataset Coastline --tracker botsort   --exp_name CoastlineBotSortGridSearch   --do_cv
# python3 evaluate_tracking_folds.py --dataset Coastline --tracker bytetrack --exp_name CoastlineByteTrackGridSearch --do_cv

# === 1.2 OpenWater Dataset === #
# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker botsort   --exp_name OpenWaterBotSortGridSearch   --do_cv
# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker bytetrack --exp_name OpenWaterByteTrackGridSearch --do_cv

# === 1.3 Merged Dataset === #
# python3 evaluate_tracking_folds.py --dataset Merged --tracker botsort   --exp_name MergedBotSortGridSearch   --do_cv
# python3 evaluate_tracking_folds.py --dataset Merged --tracker bytetrack --exp_name MergedByteTrackGridSearch --do_cv

# === 2. Test === #

# === 2.1 Coastline Dataset === #
# python3 evaluate_tracking_folds.py --dataset Coastline --tracker botsort   --exp_name CoastlineBotSortTest  
# python3 evaluate_tracking_folds.py --dataset Coastline --tracker bytetrack --exp_name CoastlineByteTrackTest 

# python3 evaluate_tracking_folds.py --dataset Coastline --coastline_model CoastlineV2 --tracker botsort   --exp_name CoastlineBotSortTestV2
# python3 evaluate_tracking_folds.py --dataset Coastline --coastline_model CoastlineV2 --tracker bytetrack --exp_name CoastlineByteTrackTestV2

# python3 evaluate_tracking_folds.py --dataset Coastline --coastline_model CoastlineV2Big --tracker botsort   --exp_name CoastlineBotSortTestV2Big
# python3 evaluate_tracking_folds.py --dataset Coastline --coastline_model CoastlineV2Big --tracker bytetrack --exp_name CoastlineByteTrackTestV2Big

# === 2.2 OpenWater Dataset === #
# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker botsort   --exp_name OpenWaterBotSortTest
# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker bytetrack --exp_name OpenWaterByteTrackTest

# === 2.3 Merged Dataset === #
# python3 evaluate_tracking_folds.py --dataset Merged --tracker botsort   --exp_name MergedBotSortTest
# python3 evaluate_tracking_folds.py --dataset Merged --tracker bytetrack --exp_name MergedByteTrackTest

# python3 evaluate_tracking_folds.py --dataset Merged --coastline_model CoastlineV2 --tracker botsort   --exp_name MergedBotSortTestV2
# python3 evaluate_tracking_folds.py --dataset Merged --coastline_model CoastlineV2 --tracker bytetrack --exp_name MergedByteTrackTestV2

# python3 evaluate_tracking_folds.py --dataset Merged --coastline_model CoastlineV2Big --tracker botsort   --exp_name MergedBotSortTestV2Big
# python3 evaluate_tracking_folds.py --dataset Merged --coastline_model CoastlineV2Big --tracker bytetrack --exp_name MergedByteTrackTestV2Big

























































# Retrieve version
# version=""

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     --version)
#       if [[ -z $2 || $2 == --* ]]; then
#         echo "Error: --version requires a value (1 or 2)."
#         exit 1
#       fi
#       version="$2"
#       shift 2
#       ;;
#     *)
#       echo "Unknown argument: $1"
#       exit 1
#       ;;
#   esac
# done

# # Check if --version was provided
# if [[ -z $version ]]; then
#   echo "Usage: $0 --version <1|2>"
#   exit 1
# fi

# # Validate version value
# if [[ "$version" != "1" && "$version" != "2" ]]; then
#   echo "Error: Version must be 1 or 2."
#   exit 1
# fi

# echo "Version specified: $version"

# #
# # Coastline Dataset
# #

# python3 evaluate_tracking_folds.py --dataset Coastline --tracker botsort $([[ "$version" == "2" ]] && echo "--v2") --exp_name CoastlineBotSortTest$([[ "$version" == "2" ]] && echo "V2Big")
# python3 evaluate_tracking_folds.py --dataset Coastline --tracker bytetrack $([[ "$version" == "2" ]] && echo "--v2") --exp_name CoastlineByteTrackTest$([[ "$version" == "2" ]] && echo "V2Big")

# #
# # OpenWater Dataset
# #

# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker botsort $([[ "$version" == "2" ]] && echo "--v2") --exp_name OpenWaterBotSortTest$([[ "$version" == "2" ]] && echo "V2Big")
# python3 evaluate_tracking_folds.py --dataset OpenWater --tracker bytetrack $([[ "$version" == "2" ]] && echo "--v2") --exp_name OpenWaterByteTrackTest$([[ "$version" == "2" ]] && echo "V2Big")

# #
# # Merged Dataset
# #

# python3 evaluate_tracking_folds.py --dataset Merged --tracker botsort $([[ "$version" == "2" ]] && echo "--v2") --exp_name MergedBotSortTest$([[ "$version" == "2" ]] && echo "V2Big")
# python3 evaluate_tracking_folds.py --dataset Merged --tracker bytetrack $([[ "$version" == "2" ]] && echo "--v2") --exp_name MergedByteTrackTest$([[ "$version" == "2" ]] && echo "V2Big")