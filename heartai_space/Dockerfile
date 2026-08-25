FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# CPU-only колесо torch — иначе pip тянет GPU-сборку с CUDA (гигабайты,
# не нужны на CPU-инстансе и раздувают память/время сборки)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
