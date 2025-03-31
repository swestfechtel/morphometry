import tempfile
import re
import logging
import os
from examination import Examination


def save_files(files: list) -> Examination:
    logger = logging.getLogger('api')
    workdir = os.path.dirname(os.path.realpath(__file__))

    with tempfile.TemporaryDirectory(dir=f'{workdir}/uploads/') as temp_dir:
        dicom_file = re.compile(r'I\d+')
        digits = re.compile(r'[1-9]')
        double_digits = re.compile(re.compile(r'[1-9]{2}'))
        single_digits = re.compile(r'I[1-9]0+$')
        multiples_of_ten = re.compile(r'I[1-9]0+1')  # TODO this will likely fail if number of files > 99
        for i, file in enumerate(files):  # have to re-order layers because pacs export has dumb file naming scheme
            filename = file.filename.split('/')[-1]
            logger.debug(filename)
            if re.match(dicom_file, filename) is None:
                logger.debug(f'Could not match {filename}')
                continue  # filter out files we do not want, e.g. VERSION files

            if re.match(single_digits, filename):  # layers no.s 1-9
                number = re.search(digits, filename)[0]
                file.save(f'{temp_dir}/I00{number}')

            elif re.match(multiples_of_ten, filename):  # layers no.s 10, 20, 30, ...
                number = re.search(digits, filename)[0]
                file.save(f'{temp_dir}/I0{number}0')

            else:  # everything else, e.g. 11-19, 21-29, ...
                number = re.search(double_digits, filename)[0]
                file.save(f'{temp_dir}/I0{number}')

        examination = Examination()
        examination.read_dicom_series(temp_dir)
        examination.read_dicom_metadata(temp_dir)
        return examination