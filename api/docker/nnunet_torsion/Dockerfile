FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN pip3 install "numpy==1.26.4"
RUN pip3 install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install "nnunetv2==2.5.1"

RUN mkdir -p /app/nnUNet_results/Dataset004_TorsionHipLarge/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres
RUN mkdir -p /app/nnUNet_results/Dataset006_TorsionKneeInternal/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres
RUN mkdir -p /app/nnUNet_results/Dataset007_TorsionAnkleInternal/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres

RUN mkdir -p /app/mnt

COPY ./checkpoints/hip/ /app/nnUNet_results/Dataset004_TorsionHipLarge/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
COPY ./checkpoints/knee/ /app/nnUNet_results/Dataset006_TorsionKneeInternal/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
COPY ./checkpoints/ankle/ /app/nnUNet_results/Dataset007_TorsionAnkleInternal/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/

COPY ./entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

RUN chown -R root:root /app && chmod -R 775 /app

ENTRYPOINT ["/app/entrypoint.sh"]