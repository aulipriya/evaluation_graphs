FROM python:3.9

WORKDIR /app

# Install dependencies needed by matplotlib/seaborn/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN chmod +x entrypoint.sh download.sh

# Run model download script during build
RUN ./download_model.sh

CMD ["./entrypoint.sh"]