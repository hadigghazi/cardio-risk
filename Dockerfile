FROM python:3.12-slim

# HOME and the matplotlib cache point somewhere writable: on the VM the container runs as the
# deploying user's uid, which has no home directory inside the image. joblib's worker pool shares
# arrays through a temporary folder: /tmp, not the container's small /dev/shm.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib JOBLIB_TEMP_FOLDER=/tmp

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "build_report.py"]
