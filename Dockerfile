FROM python:3.12-slim

# HOME and the matplotlib cache point somewhere writable: on the VM the container runs as the
# deploying user's uid, which has no home directory inside the image. joblib's worker pool shares
# arrays through a temporary folder: /tmp, not the container's small /dev/shm. No MPLBACKEND here:
# it would override the notebook kernel's inline backend and leave the report without its figures.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib JOBLIB_TEMP_FOLDER=/tmp

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "build_report.py"]
