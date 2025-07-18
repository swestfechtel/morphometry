import logging
import traceback
import tempfile
import docker
import json
import sys
import subprocess
import os

import pandas as pd
import nibabel as nib

from api.examination import TorsionExamination
from api.file_controller import FileController

from morphometry.image_io import Image, Segmentation

from pathlib import Path


class ModelJob:
    """
    Abstract class for a model job. Subclass must implement the execute method.
    """
    identifier: str
    running: bool
    file_controller: FileController
    logger: logging.Logger
    client: docker.DockerClient

    def __init__(self, file_controller: FileController):
        # self.identifier = str(uuid.uuid1())
        self.file_controller = file_controller
        self.running = False
        self.logger = logging.getLogger('api')
        self.client = docker.from_env()


class TorsionModelJob(ModelJob):

    def execute(self, examination: TorsionExamination):
        """
        Compute segmentation and torsional alignment for a TorsionExamination object.
        :param examination: A TorsionExamination object to process.
        :return: The job.
        """
        self.compute_segmentation(examination)
        self.compute_torsional_alignment(examination)

        return self

    def compute_segmentation(self, examination: TorsionExamination):
        """
        Compute segmentation masks for hip, knee and ankle.
        :param examination: A TorsionExamination object to compute segmentations for.
        :return: The job
        """
        self.logger.info(f'Created segmentation job for {examination.identifier} with identifier {self.identifier}.')
        self.running = True
        examination.status = 'running'
        self.identifier = examination.identifier  # TODO will become a problem if more than one job is running at the same time for the same examination

        examination.split_series()

        with tempfile.TemporaryDirectory() as tempdir:
            Path(f'{tempdir}/hip/input').mkdir(parents=True)
            Path(f'{tempdir}/hip/output').mkdir(parents=True)
            Path(f'{tempdir}/knee/input').mkdir(parents=True)
            Path(f'{tempdir}/knee/output').mkdir(parents=True)
            Path(f'{tempdir}/ankle/input').mkdir(parents=True)
            Path(f'{tempdir}/ankle/output').mkdir(parents=True)


            examination.hip.save_image(tempdir + '/hip/input/hip_0000.nii.gz')
            examination.knee.save_image(tempdir + '/knee/input/knee_0000.nii.gz')
            examination.ankle.save_image(tempdir + '/ankle/input/ankle_0000.nii.gz')

            docker_cmd = [
                'docker',
                'run',
                '--rm',
                '--runtime=nvidia',
                '--gpus', 'all',
                '--shm-size', '32G',
                # '--user', f'{os.getuid()}:{os.getgid()}',
                '--group-add', 'root',
                '-v', f'{tempdir}:/app/mnt:rw,Z',
                'swestfechtel/nnunet_torsion:latest'
            ]

            proc = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if proc.returncode != 0:
                self.logger.error(f"Container exited with code {proc.returncode}. Logs: {proc.stdout.decode('utf-8')}")
                raise RuntimeError(f"Segmentation job failed for {examination.identifier} with exit code {proc.returncode}.")

            self.logger.info(f"Container exited with code {proc.returncode}.")
            self.logger.debug(f'Container logs: {proc.stdout.decode("utf-8")}')

            tmp = nib.load(tempdir + '/hip/output/hip.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system(flip=False)
            examination.hip_mask = tmp

            tmp = nib.load(tempdir + '/knee/output/knee.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system(flip=False)
            examination.knee_mask = tmp

            tmp = nib.load(tempdir + '/ankle/output/ankle.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system(flip=False)
            examination.ankle_mask = tmp

        examination.encode_images()

        self.logger.info(f'Finished segmentation job for {examination.identifier} with identifier {self.identifier}.')
        examination.status = 'segmented'
        self.file_controller.update_examination(examination)

        return self

    def compute_torsional_alignment(self, examination: TorsionExamination):
        """
        Compute the torsional alignment for a TorsionExamination object.
        :param examination: A TorsionExamination object to compute torsional alignment for.
        :return: The job.
        """
        self.logger.info(f'Created torsional alignment job for {examination.identifier} with identifier {self.identifier}.')
        self.running = True
        examination.status = 'running'
        self.identifier = examination.identifier

        with tempfile.TemporaryDirectory() as tempdir:
            self.logger.debug(tempdir)
            examination.hip_mask.save_image(tempdir + '/hip_segmentation.nii.gz')
            examination.knee_mask.save_image(tempdir + '/knee_segmentation.nii.gz')
            examination.ankle_mask.save_image(tempdir + '/ankle_segmentation.nii.gz')

            docker_cmd = [
                'docker',
                'run',
                '--rm',
                '--shm-size', '32G',
                '-u', 'root',
                '-v', f'{tempdir}:/app/temp:rw,z',
                'swestfechtel/torsion:latest'
            ]

            proc = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if proc.returncode != 0:
                self.logger.error(
                    f"Container exited with code {proc.returncode}. Logs: {proc.stdout.decode('utf-8')}")
                raise RuntimeError(
                    f"Torsion job failed for {examination.identifier} with exit code {proc.returncode}.")

            self.logger.info(f"Container exited with code {proc.returncode}.")

            results = json.load(open(tempdir + '/results.json', 'r'))
            landmarks = json.load(open(tempdir + '/landmarks.json', 'r'))
            errors = json.load(open(tempdir + '/errors.json', 'r'))

            if len(errors['errors']) > 0:
                self.logger.error(f"Errors occurred during torsional alignment computation for {examination.identifier}: {errors['errors']}")
                # raise RuntimeError(f"Torsion job failed for {examination.identifier} with errors: {errors['errors']}")

            examination.femoral_torsion_left = results['femoral_torsion_left']
            examination.femoral_torsion_right = results['femoral_torsion_right']
            examination.tibial_torsion_left = results['tibial_torsion_left']
            examination.tibial_torsion_right = results['tibial_torsion_right']

            examination.landmarks = landmarks

        self.logger.info(f'Finished torsional alignment job for {examination.identifier} with identifier {self.identifier}.')

        examination.status = 'processed'
        self.file_controller.update_examination(examination)

        return self


class LandmarkJob(ModelJob):

    def __init__(self, file_controller: FileController):
        super().__init__(file_controller)

    def execute(self):
        raise NotImplementedError