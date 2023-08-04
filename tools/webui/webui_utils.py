import json

import yaml

MODEL_TYPE = {
    "vec768l12": 768,
    "vec256l9": 256,
    "hubertsoft": 256,
    "whisper-ppg": 1024,
    "cnhubertlarge": 1024,
    "dphubert": 768,
    "wavlmbase+": 768,
    "whisper-ppg-large": 1280
}
ENCODER_PRETRAIN = {
    "vec256l9": "pretrain/checkpoint_best_legacy_500.pt",
    "vec768l12": "pretrain/checkpoint_best_legacy_500.pt",
    "hubertsoft": "pretrain/hubert-soft-0d54a1f4.pt",
    "whisper-ppg": "pretrain/medium.pt",
    "cnhubertlarge": "pretrain/chinese-hubert-large-fairseq-ckpt.pt",
    "dphubert": "pretrain/DPHuBERT-sp0.75.pth",
    "wavlmbase+": "pretrain/WavLM-Base+.pt",
    "whisper-ppg-large": "pretrain/large-v2.pt"
}

class Config:
    def __init__(self, path, type):
        self.path = path
        self.type = type

    def read(self):
        if self.type == "json":
            with open(self.path, 'r') as f:
                return json.load(f)
        if self.type == "yaml":
            with open(self.path, 'r') as f:
                return yaml.safe_load(f)

    def save(self, content):
        if self.type == "json":
            with open(self.path, 'w') as f:
                json.dump(content, f, indent=4)
        if self.type == "yaml":
            with open(self.path, 'w') as f:
                yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)

