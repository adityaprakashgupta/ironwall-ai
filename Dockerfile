FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Force the low-level math library (OpenMP) to use all 8 logical threads
ENV OMP_NUM_THREADS=8

ENV NVIDIA_VISIBLE_DEVICES=all
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 2. Tell TensorFlow to split ONE operation into 8 pieces
# ENV TF_NUM_INTRAOP_THREADS=8

# 3. Tell TensorFlow to focus on one operation at a time (Sequential execution)
# ENV TF_NUM_INTEROP_THREADS=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY inference.py .
COPY config.py .
COPY data_util.py .

# Copy model
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
