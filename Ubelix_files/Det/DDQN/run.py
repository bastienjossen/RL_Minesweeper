# run.py
from config import Config
from trainer import Trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg    = Config()
Trainer(cfg, device).run(episodes=cfg.epochs)