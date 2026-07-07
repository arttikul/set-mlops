import os

import wandb

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

# Нового формату W&B Registry: колекція "emotion-classifier" у вбудованому
# реєстрі моделей "model" (не потрібен ні entity, ні активний run — див.
# training/train.py, де ран лінкує сюди нову версію артефакта).
api = wandb.Api()
artifact = api.artifact("wandb-registry-model/emotion-classifier:latest")
artifact.download(root="./model")
