import os
import sys

import sounddevice
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


from tools.infer_base import SvcInfer

svc: SvcInfer = SvcInfer()

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

sampling_rate, audio = res
sounddevice.play(audio, sampling_rate, blocking=True)
