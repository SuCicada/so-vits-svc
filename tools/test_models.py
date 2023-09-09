import os
import sys

import sounddevice
# import sounddevice
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools.file_util import get_root_project
from tools.infer import models_info
from tools import infer_base
from tools.infer_base import SvcInfer

model_info = models_info.LainV2
config = model_info.to_svc_config(f"{get_root_project()}/models")
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
    # svc = svcInfer.
    audio = "tools/trash/1_Bôa - Duvet TV Sized_(Vocals).wav"
    target_sampling_rate, target_audio = svcInfer.transform_audio_file(audio,
                                                                       # enhancer_adaptive_key=8
                                                                       )
    soundfile.write("out.wav", target_audio, target_sampling_rate, format="wav")
    sounddevice.play(target_audio, target_sampling_rate, blocking=True)


# tts()

svc()
