from transformers.generation.streamers import BaseStreamer
from tqdm import tqdm
import torch


class TokenProgressStreamer(BaseStreamer):
    def __init__(self, max_new_tokens: int):
        self.pbar = tqdm(total=max_new_tokens, desc='Generating', unit=' tok')
        self.started = False

    def put(self, value: torch.Tensor):
        if self.started:
            self.pbar.update(1)
        self.started = True

    def end(self):
        self.pbar.close()
