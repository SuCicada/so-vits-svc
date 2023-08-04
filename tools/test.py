import os
import sys

import sounddevice
# import sounddevice
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools import infer_base
from tools.infer_base import SvcInfer

config = {
    "model_path": "models/G_2400_infer.pth",
    "config_path": "models/config.json",
    "cluster_model_path": "models/kmeans_10000.pt",
    "diff_model_path": "models/diffusion/model_12000.pt",
    "diff_config_path": "models/diffusion/config.yaml",
}
svcInfer: SvcInfer = SvcInfer(config)


# print(sounddevice.query_devices())
def tts():
    res = svcInfer.get_audio(
        # tts_engine="edge-tts",
        tts_engine="gtts",
        text="""
        人はみな、繋がっている。
        """,
        language="ja",
        speed=1.1,

        # text="我在哪里，你是谁",
        # language="zh",
    )

    sampling_rate, audio = res[1]
    soundfile.write("out.wav", audio, sampling_rate, format="wav")
    sounddevice.play(audio, sampling_rate, blocking=True)


def svc():
    svc = svcInfer.svc

    audio = "1_Bôa - Duvet TV Sized_(Vocals).wav"
    target_sampling_rate, target_audio = infer_base.transform_audio(audio, svc,
                                                                    enhancer_adaptive_key=8)
    soundfile.write("out.wav", target_audio, target_sampling_rate, format="wav")


tts()
