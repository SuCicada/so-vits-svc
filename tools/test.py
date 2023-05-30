import os
import sys

# import sounddevice
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools import infer_base
from tools.infer_base import SvcInfer



svcInfer: SvcInfer = SvcInfer(
    model_path="models/G_256800_infer.pth",
    config_path="models/config.json",
    cluster_model_path="models/kmeans_10000.pt",
    hubert_model_path="models/checkpoint_best_legacy_500.pt"
)

# print(sounddevice.query_devices())
def tts():
    res = svcInfer.get_audio(
        # tts_engine="edge-tts",
        tts_engine="gtts",
        text="私はどこ? あなたは誰? ",
        language="ja",
        speed=1.1,

        # text="我在哪里，你是谁",
        # language="zh",
    )

    sampling_rate, audio = res[1]
    # sounddevice.play(audio, sampling_rate, blocking=True)
    soundfile.write("out.wav", audio, sampling_rate, format="wav")
def svc():
    svc = svcInfer.svc

    audio = "1_Bôa - Duvet TV Sized_(Vocals).wav"
    target_sampling_rate, target_audio = infer_base.transform_audio(audio, svc,
                                                                    enhancer_adaptive_key=8)
    soundfile.write("out.wav", target_audio, target_sampling_rate, format="wav")

tts()
