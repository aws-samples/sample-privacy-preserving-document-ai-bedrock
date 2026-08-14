FROM public.ecr.aws/docker/library/python:3.12-slim

# poppler-utils: pdf2image shells out to `pdftoppm`. libgomp1: PaddlePaddle's
# OpenMP runtime dependency on slim Debian images. libgl1/libglib2.0-0: PaddleOCR
# pulls in full (non-headless) opencv transitively, which dlopen()s libGL at
# import time even though nothing here does any GUI/display work.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8083
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8083"]
