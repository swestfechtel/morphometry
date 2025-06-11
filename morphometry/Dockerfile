from python:3.10

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir nibabel matplotlib numpy scipy pydicom SimpleITK pandas pyvista scikit-image scikit-learn

RUN mkdir -p /app/morphometry

COPY /home/simon/Work/morphometry/morphometry/__init__.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/ankle.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/bresenham.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/femur.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/hip.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/image_io.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/knee.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/tibia.py /app/morphometry/
COPY /home/simon/Work/morphometry/morphometry/utils.py /app/morphometry/

COPY ./compute_torsion.py /app/

CMD ["python3", "/app/compute_torsion.py"]