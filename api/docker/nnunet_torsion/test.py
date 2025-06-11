import shutil
import tempfile
import os
import subprocess
import sys

if __name__ == '__main__':
    container_image = 'swestfechtel/nnunet_torsion:latest'

    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy('/home/simon/Downloads/deckers/hip.nii.gz', temp_dir + '/hip.nii.gz')
        shutil.copy('/home/simon/Downloads/deckers/knee.nii.gz', temp_dir + '/knee.nii.gz')
        shutil.copy('/home/simon/Downloads/deckers/ankle.nii.gz', temp_dir + '/ankle.nii.gz')

        docker_cmd = [
            'docker',
            'run',
            '--rm',
            '--runtime=nvidia',
            '--gpus', 'all',
            '--shm-size', '32G',
            '-v', f'{temp_dir}:/app/temp:rw',
            # '-u', f'{os.getuid()}:{os.getgid()}',  # <-- Remove this line
            container_image
        ]

        try:
            proc = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            print(f"Container exited with code {proc.returncode}")
            print("Container output:")
            print(proc.stdout.decode('utf-8'))

            if proc.returncode == 0:
                shutil.copy(f'{temp_dir}/hip.nii.gz', '/home/simon/Downloads/deckers/hip_segmentation.nii.gz')
                shutil.copy(f'{temp_dir}/knee.nii.gz', '/home/simon/Downloads/deckers/knee_segmentation.nii.gz')
                shutil.copy(f'{temp_dir}/ankle.nii.gz', '/home/simon/Downloads/deckers/ankle_segmentation.nii.gz')
        except Exception as e:
            print(f"Error running docker: {e}", file=sys.stderr)
