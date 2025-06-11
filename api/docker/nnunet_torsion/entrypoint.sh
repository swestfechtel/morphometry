#!/usr/bin/env bash

export nnUNet_raw="/app/nnUNet_raw_data"
export nnUNet_preprocessed="/app/"
export nnUNet_results="/app/nnUNet_results"

export MKL_SERVICE_FORCE_INTEL=1

echo "Creating directories..."
mkdir -p /app/temp/hip/input
mkdir -p /app/temp/hip/output
mkdir -p /app/temp/knee/input
mkdir -p /app/temp/knee/output
mkdir -p /app/temp/ankle/input
mkdir -p /app/temp/ankle/output

echo "/temp/ contents:"
ls -l /app/temp/

echo "Moving input files to input directories..."

cp /app/temp/hip.nii.gz /app/temp/hip/input/hip_0000.nii.gz
cp /app/temp/knee.nii.gz /app/temp/knee/input/knee_0000.nii.gz
cp /app/temp/ankle.nii.gz /app/temp/ankle/input/ankle_0000.nii.gz

ls -l /app/temp/hip/input/
ls -l /app/temp/knee/input/
ls -l /app/temp/ankle/input/

# python3 predict.py
nnUNetv2_predict -i /app/temp/hip/input -o /app/temp/hip/output -d 8 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda
nnUNetv2_predict -i /app/temp/knee/input -o /app/temp/knee/output -d 21 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda
nnUNetv2_predict -i /app/temp/ankle/input -o /app/temp/ankle/output -d 22 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda

echo "Prediction completed successfully. Moving files to /app/"

cp /app/temp/hip/output/* /app/temp/
cp /app/temp/knee/output/* /app/temp/
cp /app/temp/ankle/output/* /app/temp/

ls -l /app/temp/