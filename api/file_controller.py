import tempfile
import re
import logging
import os
import string
import random
import pickle
import shutil
import json

from api.examination import Examination

from morphometry.image_io import Image

from queue import Queue
from copy import deepcopy
from pathlib import Path
from fastapi import UploadFile


class FileController(object):
    queue: Queue
    cache: dict
    storage_dir: str
    stored_files: set
    logger: logging.Logger

    def __init__(self, cache_size: int = 100, storage_dir: str = 'data'):
        """
        Instantiate a new FileController.
        :param cache_size: Maximum number of examinations that can be stored in the cache. Default is 100.
        """
        self.queue = Queue(maxsize=cache_size)
        self.cache = dict()
        workdir = os.path.dirname(os.path.realpath(__file__))
        self.storage_dir = f'{workdir}/{storage_dir}'
        self._load_stored_files()
        self.logger = logging.getLogger('api')

    def copy(self):
        """
        Create a deep copy of the examination.
        :return: A deep copy of the examination.
        """
        return deepcopy(self)

    def _add_to_cache(self, examination: Examination):
        """
        Add an examination to the cache.
        :param examination: An Examination object to add.
        :return:
        """
        if self.queue.full():
            id_to_remove = self.queue.get()
            examination_to_remove = self.cache[id_to_remove]
            self._save_examination_to_disk(examination_to_remove)  # make sure to save the examination to disk when we remove it from the cache
            del self.cache[id_to_remove]

        self.queue.put(examination.identifier)
        self.cache[examination.identifier] = examination

    def _load_stored_files(self):
        """
        Load file identifiers from storage directory.
        :return:
        """
        file_identifiers = os.listdir(self.storage_dir)
        file_identifiers = [f.split('.')[0] for f in file_identifiers]
        self.stored_files = set(file_identifiers)

    def _save_examination_to_disk(self, examination: Examination):
        """
        Save an examination to disk.
        :param examination: An Examination object to save.
        :return:
        """
        with open(f'{self.storage_dir}/{examination.identifier}.pkl', 'wb') as f:
            pickle.dump(examination, f)

            self._load_stored_files()  # make sure to update the stored files after saving
            self.logger.info(f'Saved examination {examination.identifier} to disk.')

    def _load_examination_from_disk(self, identifier: str) -> Examination | None:
        """
        Load an examination from disk.
        :param identifier: The identifier of the examination to load.
        :return: The loaded Examination object.
        """

        try:
            with open(f'{self.storage_dir}/{identifier}.pkl', 'rb') as f:
                examination = pickle.load(f)

                self.logger.info(f'Loaded examination {identifier} from disk.')
                self._add_to_cache(examination)

                return examination

        except FileNotFoundError:
            self.logger.error(f'Examination {identifier} not found on disk.')

            return None

    def add_examination(self, examination: Examination) -> Examination | bool:
        """
        Add a new examination.
        :param examination: An Examination object to add.
        :return: The added Examination object or False if the examination already exists.
        """
        """
        id_length = 10  # length of the session id
        letters = string.ascii_lowercase
        random_file_id = ''.join(random.choice(letters) for i in range(id_length))
        while random_file_id in self.stored_files:
            random_file_id = ''.join(random.choice(letters) for i in range(id_length))
        """
        identifier = examination.metadata[0x0008, 0x0050].value  # accession number

        if identifier in self.stored_files:
            # return False
            pass  # TODO handle duplicate uploads

        self.stored_files.add(identifier)
        examination.identifier = identifier
        self._save_examination_to_disk(examination)
        self._add_to_cache(examination)

        return examination

    def update_examination(self, examination: Examination):
        """
        Update an existing examination.

        Saves the examination to disk and updates the cache.
        :param examination: An Examination object to update.
        :return:
        """

        if examination.identifier not in self.stored_files:  # if the examination was not saved to disk yet
            self.add_examination(examination)
            return  # don't need to return the reference here, just end

        if examination.identifier not in self.cache.keys(): # if the examination somehow was removed from the cache during runtime
            self._add_to_cache(examination)

        self.cache[examination.identifier] = examination
        self.logger.info(f'Updated examination {examination.identifier} in cache.')

        self._save_examination_to_disk(examination)
        self.logger.info(f'Updated examination {examination.identifier} on disk.')

    def get_examination(self, identifier: str) -> Examination:
        """
        Retrieve an examination from the cache or disk.
        :param identifier: The identifier of the examination to retrieve.
        :return: The retrieved Examination object.
        """
        if identifier in self.cache:
            self.logger.info(f'Retrieved examination {identifier} from cache.')
            return self.cache[identifier]

        self.logger.info(f'Retrieved examination {identifier} from disk.')
        return self._load_examination_from_disk(identifier)

    def get_examinations(self) -> list[str]:
        """
        Retrieve a list of all examinations saved on disk.
        :return: A list of examination identifiers.
        """
        return list(self.stored_files)

    def delete_examination(self, identifier: str) -> bool:
        """
        Delete an examination from the cache and disk.
        :param identifier: The identifier of the examination to delete.
        :return: True if the examination was deleted, False otherwise.
        """

        if identifier in self.cache.keys():
            del self.cache[identifier]  # this only deletes the reference to the examination in the cache, not the object itself!

            tmp = list()
            while not self.queue.empty():  # empty the queue
                tmp.append(self.queue.get())

            tmp.remove(identifier)  # remove the identifier from the queue

            for item in tmp:  # rebuild the queue
                self.queue.put(item)

            self.logger.info(f'Deleted examination {identifier} from cache.')

        try:
            os.remove(f'{self.storage_dir}/{identifier}.pkl')
            self.logger.info(f'Deleted examination {identifier} from disk.')
            self.stored_files.remove(identifier)

            return True

        except FileNotFoundError:
            self.logger.error(f'Examination {identifier} not found on disk.')

            return False

    @staticmethod
    def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
        try:
            with destination.open("wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
        finally:
            upload_file.file.close()

    @staticmethod
    def filter_orthanc_files(files: list[UploadFile]) -> list[UploadFile]:
        """
        Filter files received from Orthanc according to pre-specified rules.
        :param files: A list of dicom files received from an orthanc server.
        :return:
        """
        raise NotImplementedError
        with open('filter_rules.json') as f:
            filter_rules = json.load(f)

        ds = pydicom.dcm


    def save_files(self, files: list[UploadFile], origin: str) -> Examination | bool:
        """
        Save uploaded files to a temporary directory and create an Examination object.
        :param files: A list of uploaded files.
        :param origin: The origin of the files (e.g. "orthanc").
        :return: The created Examination object or False if the examination already exists.
        """
        workdir = os.path.dirname(os.path.realpath(__file__))
        self.logger.debug(f'Got files {files}')

        with (tempfile.TemporaryDirectory(dir=f'{workdir}/uploads/') as temp_dir):
            dicom_file = re.compile(r'I\d+')
            digits = re.compile(r'[1-9]')
            double_digits = re.compile(re.compile(r'[1-9]{2}'))
            single_digits = re.compile(r'I[1-9]0+$')
            multiples_of_ten = re.compile(r'I[1-9]0+1')  # TODO this will likely fail if number of files > 99
            for i, file in enumerate(files):  # have to re-order layers because pacs export has dumb file naming scheme
                if origin == 'ui':  # if files were sent via the ui

                    filename = file.filename.split('/')[-1]
                    self.logger.debug(filename)
                    if re.match(dicom_file, filename) is None:
                        self.logger.debug(f'Could not match {filename}')
                        continue  # filter out files we do not want, e.g. VERSION files

                    if re.match(single_digits, filename):  # layers no.s 1-9
                        number = re.search(digits, filename)[0]
                        self.save_upload_file(file, Path(f'{temp_dir}/I00{number}'))

                    elif re.match(multiples_of_ten, filename):  # layers no.s 10, 20, 30, ...
                        number = re.search(digits, filename)[0]
                        self.save_upload_file(file, Path(f'{temp_dir}/I0{number}0'))

                    else:  # everything else, e.g. 11-19, 21-29, ...
                        number = re.search(double_digits, filename)[0]
                        self.save_upload_file(file, Path(f'{temp_dir}/I0{number}'))

                elif origin == 'orthanc':  # if files were sent from orthanc, they are already in the correct order
                    self.save_upload_file(file, Path(f'{temp_dir}/{file.filename}'))  # so just save them as is

            metdata = Image.read_dicom_metadata(temp_dir)

            nib_image, tmp = Image.dicom_to_nibabel(temp_dir)

            self.logger.debug(f'Loaded image with shape {nib_image.get_fdata().shape}')

            image = Image.from_nibabel(nib_image)

            image.metadata = metdata

            transformed_image = image.copy()
            transformed_image.transform_coordinate_system()

            examination = Examination(identifier=None, original_image=image, transformed_image=transformed_image, metadata=metdata)
            tmp.cleanup()

            return self.add_examination(examination)  # if this returns False the examination object should be garbage collected
