import docker
import shutil
import tempfile
import os

if __name__ == '__main__':
    client = docker.from_env()
    container_image = 'swestfechtel/torsion:latest'

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the input file into the temp directory
        shutil.copy('/home/simon/Downloads/deckers/hip_segmentation.nii.gz', temp_dir)
        shutil.copy('/home/simon/Downloads/deckers/knee_segmentation.nii.gz', temp_dir)
        shutil.copy('/home/simon/Downloads/deckers/ankle_segmentation.nii.gz', temp_dir)

        # Run the container with the temp directory mounted
        container = client.containers.run(
            container_image,
            volumes={temp_dir: {'bind': '/app/temp', 'mode': 'rw'}},
            detach=True,
        )

        try:
            # Wait for the container to finish
            result = container.wait()
            exit_code = result.get('StatusCode', -1)
            logs = container.logs(stdout=True, stderr=True)
            print(f"Container exited with code {exit_code}")
            print("Container output:")
            print(logs.decode('utf-8'))

            if exit_code == 0:
                shutil.copy(f'{temp_dir}/results.json', '/home/simon/Downloads/deckers/')
                shutil.copy(f'{temp_dir}/landmarks.json', '/home/simon/Downloads/deckers/')
                shutil.copy(f'{temp_dir}/errors.json', '/home/simon/Downloads/deckers/')
        finally:
            container.remove()
            client.close()
