# Базовий образ: легкий офіційний python, torch ставимо окремо у CPU-версії
# (аналогічно 4-Inference/FastAPI_Docker з лекції).
FROM python:3.11-slim

ENV WANDB_CACHE_DIR=/app/.cache/wandb
WORKDIR /app
RUN mkdir -p /app/.cache/wandb /app/model

# Копіюємо файли
COPY requirements.txt requirements.txt
COPY download_model.py download_model.py
COPY main.py main.py
COPY start.sh start.sh
RUN chmod +x start.sh

# CPU-версія torch (легша за дефолтну CUDA-збірку), далі — решта залежностей
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["./start.sh"]
