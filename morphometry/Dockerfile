FROM python:3.10

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir nibabel matplotlib numpy scipy pydicom SimpleITK pandas pyvista scikit-image scikit-learn Deprecated

# Copy the whole morphometry package (region modules + measurements/ + cartilage/
# + constants.py / geometry.py / whole_leg.py). The build context is this dir.
COPY . /app/morphometry/

# Entry point runs from /app so `import morphometry.*` resolves /app/morphometry.
COPY ./docker/compute_torsion.py /app/

RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

CMD ["python3", "/app/compute_torsion.py"]