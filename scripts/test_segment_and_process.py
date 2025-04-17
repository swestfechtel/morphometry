import argparse
import requests
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--accession_number', type=str, required=True, help='Accession number of the examination to process')
    parser.add_argument('-s', '--segment_only', action='store_true', help='Only segment the examination')
    args = parser.parse_args()

    url = ('http://localhost:8000/model/torsion/' if not args.segment_only else 'http://localhost:8000/model/segmentation/') + args.accession_number

    response = requests.post(url)
    if response.status_code == 202:
        print(f'Request accepted, processing job {response.text} started.')

        job_id = response.json().get('job_id')
        while True:
            job_response = requests.get(f'http://localhost:8000/jobs/{job_id}')
            if job_response.status_code == 200:
                status = job_response.json().get('status')
                if status == 'finished':
                    print('Job finished.')
                    break
                else:
                    print('Job is still running...')
                    time.sleep(5)
            else:
                print('Error retrieving job status:', job_response.status_code)
                break

    elif response.status_code == 400:
        print('Bad request.')
    elif response.status_code == 404:
        print('Examination not found.')
    else:
        print('Unknown error:', response.status_code)