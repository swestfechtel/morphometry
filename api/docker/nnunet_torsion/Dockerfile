# nnUNet segmentation image (hip / knee / ankle, ResEncUNetXL, 5-fold ensemble).
#
# Size notes (the pre-1.0.0 image was 79.5 GB, this one 25.6 GB):
#  - Checkpoints are copied with COPY --chmod. A later `RUN chown/chmod -R` would rewrite every
#    file into a new layer and store all checkpoints a second time.
#  - Only checkpoint_final.pth is loaded (entrypoint.sh passes no -chk); .dockerignore keeps
#    checkpoint_best.pth, training logs and plots out of the build context.
#  - pip runs with --no-cache-dir so no download cache is left in the image.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "pip==25.1.1"

# The exact package set of the pre-versioning image, installed without dependency resolution so a
# missing or drifting package fails the build instead of resolving to whatever is newest.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt \
    && pip3 check \
    && rm /tmp/requirements.txt

ARG RESULTS=/app/nnUNet_results
ARG TRAINER=nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres

RUN mkdir -p /app/mnt

COPY --chmod=775 ./checkpoints/hip/ ${RESULTS}/Dataset004_TorsionHipLarge/${TRAINER}/
COPY --chmod=775 ./checkpoints/knee/ ${RESULTS}/Dataset006_TorsionKneeInternal/${TRAINER}/
COPY --chmod=775 ./checkpoints/ankle/ ${RESULTS}/Dataset007_TorsionAnkleInternal/${TRAINER}/

COPY --chmod=775 ./entrypoint.sh /app/

ENTRYPOINT ["/app/entrypoint.sh"]
