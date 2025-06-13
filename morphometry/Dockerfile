FROM python:3.10

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir nibabel matplotlib numpy scipy pydicom SimpleITK pandas pyvista scikit-image scikit-learn Deprecated

RUN mkdir -p /app/morphometry

COPY ./__init__.py /app/morphometry/
COPY ./ankle.py /app/morphometry/
COPY ./bresenham.py /app/morphometry/
COPY ./femur.py /app/morphometry/
COPY ./hip.py /app/morphometry/
COPY ./image_io.py /app/morphometry/
COPY ./knee.py /app/morphometry/
COPY ./tibia.py /app/morphometry/
COPY ./utils.py /app/morphometry/

COPY ./docker/compute_torsion.py /app/

RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

CMD ["python3", "/app/compute_torsion.py"]