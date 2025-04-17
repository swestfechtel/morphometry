import argparse
import requests
import pprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--accession_number', type=str, required=True, help='Accession number of the examination to process')
    args = parser.parse_args()

    url = 'http://localhost:8000/examinations/' + args.accession_number

    response = requests.get(url)
    if response.status_code == 200:
        print('Examination found:')
        pprint.pprint(response.json())
    elif response.status_code == 404:
        print('Examination not found.')
    else:
        print('Unknown error:', response.status_code)
