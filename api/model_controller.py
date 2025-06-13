import logging
import traceback
import tempfile
import docker
import json
import sys
import subprocess

import pandas as pd
import nibabel as nib

from api.examination import TorsionExamination
from api.file_controller import FileController

from morphometry.image_io import Image, Segmentation


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
            examination.hip.save_image(tempdir + '/hip.nii.gz')
            examination.knee.save_image(tempdir + '/knee.nii.gz')
            examination.ankle.save_image(tempdir + '/ankle.nii.gz')

            docker_cmd = [
                'docker',
                'run',
                '--rm',
                '--runtime=nvidia',
                '--gpus', 'all',
                '--shm-size', '32G',
                '-v', f'{tempdir}:/app/temp:rw',
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

            tmp = nib.load(tempdir + '/hip.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system()
            examination.hip_mask = tmp

            tmp = nib.load(tempdir + '/knee.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system()
            examination.knee_mask = tmp

            tmp = nib.load(tempdir + '/ankle.nii.gz')
            tmp = Segmentation.from_nibabel(tmp)
            tmp.transform_coordinate_system()
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
            examination.hip_mask.save_image(tempdir + '/hip_segmentation.nii.gz')
            examination.knee_mask.save_image(tempdir + '/knee_segmentation.nii.gz')
            examination.ankle_mask.save_image(tempdir + '/ankle_segmentation.nii.gz')

            container = self.client.containers.run(
                'swestfechtel/torsion:latest',
                volumes={tempdir: {'bind': '/app/temp', 'mode': 'rw'}},
                detach=True
            )

            try:
                result = container.wait()
                exit_code = result.get('StatusCode', -1)

                if exit_code != 0:
                    logs = container.logs(stdout=True, stderr=True)
                    self.logger.error(f"Container exited with code {exit_code}. Logs: {logs.decode('utf-8')}")
                    raise RuntimeError(f"Torsional alignment job failed for {examination.identifier} with exit code {exit_code}.")

                self.logger.info(f"Container exited with code {exit_code}.")

                results = json.load(open(tempdir + '/results.json', 'r'))
                landmarks = json.load(open(tempdir + '/landmarks.json', 'r'))
                errors = json.load(open(tempdir + '/errors.json', 'r'))

                examination.femoral_torsion_left = results['femoral_torsion_left']
                examination.femoral_torsion_right = results['femoral_torsion_right']
                examination.tibial_torsion_left = results['tibial_torsion_left']
                examination.tibial_torsion_right = results['tibial_torsion_right']

                examination.landmarks = landmarks

            finally:
                container.remove()

        self.logger.info(f'Finished torsional alignment job for {examination.identifier} with identifier {self.identifier}.')

        examination.status = 'processed'
        self.file_controller.update_examination(examination)

        return self


class LandmarkJob(ModelJob):

    def __init__(self, file_controller: FileController):
        super().__init__(file_controller)

    def execute(self):
        raise NotImplementedError
