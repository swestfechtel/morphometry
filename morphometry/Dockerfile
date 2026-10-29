# Pinned to the venv's exact interpreter (3.10.16), not the floating `python:3.10`, which had
# already drifted to 3.10.22 on Debian 13 / GCC 14.2.0 while the venv is 3.10.16 / GCC 11.2.0.
# The 38 pins in requirements.txt only fix the Python packages; the interpreter, libc and
# compiler are fixed here, and without this line "the image and the venv are the same code path"
# is not true below the wheels. Bookworm because that is what python:3.10.16 ships.
# Residual, deliberately not closed: this interpreter is built with GCC 12.2.0 against glibc 2.36
# and the venv's with GCC 11.2.0 against glibc 2.43, and the base is pinned by tag, not digest.
# It changes nothing on the Augsburg fixture (the rebuild is byte-identical to the one on the
# floating base), partly because numpy==1.26.4 is the same OpenBLAS DYNAMIC_ARCH manylinux wheel
# either way and the same CPU dispatches the same kernels -- not guaranteed on another host.
FROM python:3.10.16-bookworm

WORKDIR /app

# The shared libraries vtk links at import time. `measurements/hip.py` annotates
# `plot: plt.Axes | pv.Plotter | bool`, which is evaluated at def time, so merely importing the
# measurements pulls in `pyvista.plotting` -> `vtkmodules.vtkInteractionWidgets` -> libGL.
# Nothing in this image renders anything; these are needed only to make that import succeed.
# (The unpinned builds happened to resolve a pyvista/vtk pair that did not need them, so this
# line is part of pinning pyvista to the venv's 0.44.1 / vtk 9.3.1.)
# Kept minimal on purpose. `objdump -p` over every .so in site-packages gives exactly three
# system X/GL libraries as NEEDED -- libGL.so.1, libX11.so.6, libXrender.so.1 -- plus vendored,
# hash-suffixed copies vtk's wheel ships itself (libXau, libXcursor, libXfixes, libxcb; they read
# as "not found" under a bare ldd and resolve at runtime through the wheel's RPATH). There is
# NOTHING from glib, gobject, gthread, libXext, libSM or libICE anywhere in site-packages, so the
# `libglib2.0-0 libxext6` this line once also carried were dead weight -- and on Debian 13
# `libglib2.0-0` is only a Provides alias of `libglib2.0-0t64`, a name a later base can drop.
# libxrender1 is NOT transitively pulled by libgl1 on bookworm (its installed reverse-deps are
# libcairo2 / libxft2 / libxrender-dev), so it has to be named: it is load-bearing, not defence.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies from an explicit, fully pinned requirement file rather than from an
# unpinned `pip install a b c` line. The versions in requirements.txt are exactly the ones the
# repo-root requirements.txt (the research venv the measurements are developed and golden-tested
# in) holds, so the image and the venv are the same code path. The unpinned line this replaced
# resolved NumPy 2 while the venv pins numpy==1.26.4. Copied before the package so that editing
# the library does not invalidate the (slow) dependency layer.
#
# NOTE: this image computes measurements, so it must be REBUILT after every landmark /
# measurement change in morphometry/ -- otherwise the API keeps serving the numbers of whatever
# commit the image was last built from. See the "Docker images" section of CLAUDE.md.
# --no-deps: requirements.txt holds the *complete* transitive closure, so resolving dependencies
# again can only pull something unpinned. With --no-deps a package missing from the closure is a
# build-time failure instead of a silent unpinned install (`pip check` in the built image
# confirms the closure is satisfied).
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir --no-deps -r /app/requirements.txt \
    && pip3 check

# Copy the whole morphometry package (region modules + measurements/ + cartilage/
# + constants.py / geometry.py / whole_leg.py). The build context is this dir.
COPY . /app/morphometry/

# Entry point runs from /app so `import morphometry.*` resolves /app/morphometry.
COPY ./docker/compute_torsion.py /app/

RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

CMD ["python3", "/app/compute_torsion.py"]
