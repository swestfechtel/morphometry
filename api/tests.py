from api.file_controller import FileController
from api.model_controller import ModelJob
from api.examination import TorsionExamination
from dataclasses import dataclass
from typing import AnyStr
from pathlib import Path
import requests
import time
import asyncio

@dataclass
class PseudoFile:
    filename: str
    file: AnyStr

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.file = f.read()

        self.filename = path.name

    def save(self, path: str):
        with open(path, 'wb') as f:
            f.write(self.file)


def test_file_upload():
    files = list()
    for file in Path('/home/simon/Downloads/deckers/3stack').iterdir():
        if file.is_file():
            files.append(PseudoFile(file))

    response = requests.post('http://localhost:8000/upload/', files=files)
    return response


def test_segmentation(id: str) -> str:
    file_controller = FileController()
    examination = file_controller._load_examination_from_disk(id)

    torsion_examination = TorsionExamination(examination)
    torsion_examination.split_series()

    ModelJob.compute_segmentation(torsion_examination)

    file_controller._save_examination_to_disk(torsion_examination)
    return torsion_examination.identifier


def test_torsion_computation(id: str) -> None:
    file_controller = FileController()
    examination = file_controller._load_examination_from_disk(id)

    ModelJob.compute_torsional_alignment(examination)

    print(examination.get_torsion_values())
    print(examination.landmarks)


async def f():
    time.sleep(5)
    print('f')


async def main():
    print('start')
    # asyncio.create_task(f())
    await f()
    print('end')
    return 1


if __name__ == '__main__':
    print(asyncio.run(main()))
