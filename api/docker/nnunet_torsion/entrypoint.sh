#!/usr/bin/env bash

export nnUNet_raw="/app/nnUNet_raw_data"
export nnUNet_preprocessed="/app/"
export nnUNet_results="/app/nnUNet_results"

export MKL_SERVICE_FORCE_INTEL=1

echo "/app/mnt contents:"
ls -l /app/mnt

echo "nnUNet results directories contents:"
ls -l "/app/nnUNet_results/Dataset007_TorsionAnkleInternal/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"

echo "Input directories contents:"
ls -l /app/mnt/hip/input/
ls -l /app/mnt/knee/input/
ls -l /app/mnt/ankle/input/

nnUNetv2_predict -i /app/mnt/hip/input -o /app/mnt/hip/output -d 4 -c 3d_fullres -p nnUNetResEncUNetXLPlans -f 0 1 2 3 4 -device cuda --disable_tta
nnUNetv2_predict -i /app/mnt/knee/input -o /app/mnt/knee/output -d 6 -c 3d_fullres -p nnUNetResEncUNetXLPlans -f 0 1 2 3 4 -device cuda --disable_tta
nnUNetv2_predict -i /app/mnt/ankle/input -o /app/mnt/ankle/output -d 7 -c 3d_fullres -p nnUNetResEncUNetXLPlans -f 0 1 2 3 4 -device cuda --disable_tta

echo "Prediction completed for hip, knee, and ankle datasets."