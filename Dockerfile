FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    diffusers==0.36.0 \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    "imageio[ffmpeg]==2.37.0" \
        sentencepiece==0.2.0 \
    ftfy==6.3.1

COPY handler.py /app/handler.py

ENV HF_HOME=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/handler.py"]
# Trigger build after reconnect
