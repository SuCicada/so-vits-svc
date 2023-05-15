import os
import sys

import sounddevice
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


from tools.infer_base import SvcInfer

svc: SvcInfer = SvcInfer(
    model_path="/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/G_256800_infer.pth",
    config_path="/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/config.json",
    cluster_model_path="/Users/peng/PROGRAM/GitHub/so-vits-svc/logs/lain/kmeans_10000.pt",
)

print(sounddevice.query_devices())

res = svc.get_audio(
    # tts_engine="edge-tts",
    tts_engine="gtts",
    text="私はどこ? あなたは誰? ",
    language="ja",
    speed=1.0,

    # text="我在哪里，你是谁",
    # language="zh",
)

sampling_rate, audio = res[1]
# sounddevice.play(audio, sampling_rate, blocking=True)
soundfile.write("out.wav", audio, sampling_rate, format="wav")
