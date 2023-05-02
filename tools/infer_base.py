import io
import os
import tempfile
from io import BytesIO

import librosa
import numpy
import numpy as np
import sounddevice
import soundfile
import torch
from gtts import gTTS
from pydub import AudioSegment

from inference.infer_tool import Svc
import logging

from . import audio_utils, tts_utils
from .audio_utils import modify_speed

logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('gtts').setLevel(logging.INFO)

auto_f0 = True  # "自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）"
vc_transform = 0  # 变调（整数，可以正负，半音数量，升高八度就是12

pad_seconds = 0.5  # 推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现
cl_num = 0  # 音频自动切片，0为不切片，单位为秒(s)
lg_num = 0  # "两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s
lgr_num = 0.75  # 自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭

# not suggest to modify
sid = "lain"
slice_db = -40
noice_scale = 0.4

enhancer_adaptive_key = 0  # "使增强器适应更高的音域(单位为半音数)|默认为0"
cr_threshold = 0.05  # "F0过滤阈值，只有启动f0_mean_pooling时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音"
F0_mean_pooling = False  # 是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭
cluster_ratio = 0  # 聚类模型混合比例，0-1之间，0即不启用聚类。使用聚类模型能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）


class SvcInfer:
    model_path = "/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/G_256800_infer.pth"
    config_path = "/Users/peng/PROGRAM/GitHub/so-vits-svc/lain/config.json"

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svc: Svc = Svc(net_g_path=self.model_path,
                            config_path=self.config_path,
                            device=device,
                            cluster_model_path="",
                            nsf_hifigan_enhance=False)

    def get_audio(self, text,
                  tts_engine="edge-tts",
                  language="ja",
                  speed=1.0,
                  ) -> ((int, np.array), (int, np.array)):
        sampling_rate, audio = tts_utils.text_to_audio(text, tts_engine, language)
        # sounddevice.play(audio, sampling_rate, blocking=True)
        origin_sampling_rate, origin_audio = sampling_rate, audio

        target_sampling_rate, target_audio = self.transform_audio(sampling_rate, audio)
        if speed != 1.0:
            target_audio = modify_speed(target_audio, sampling_rate, speed)
        return (origin_sampling_rate, origin_audio), (target_sampling_rate, target_audio)

    def transform_audio(self, sampling_rate, audio) -> (int, np.array):
        audio: np.array
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # temp_path = "temp.wav"
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
            temp_path = temp_file.name
            soundfile.write(temp_path, audio, sampling_rate, format="wav")
            # os.remove(temp_path)
            target_sampling_rate, target_audio = transform_audio(temp_path, svc=self.svc)

            return (target_sampling_rate, target_audio)


def transform_audio(audio_path, svc: Svc) -> (int, np.array):
    _audio: np.array = svc.slice_inference(raw_audio_path=audio_path,
                                           spk=sid,
                                           tran=vc_transform,
                                           slice_db=slice_db,
                                           cluster_infer_ratio=cluster_ratio,
                                           auto_predict_f0=auto_f0,
                                           noice_scale=noice_scale,
                                           # pad_seconds,
                                           # cl_num, lg_num, lgr_num,
                                           # F0_mean_pooling,
                                           # enhancer_adaptive_key,
                                           # cr_threshold
                                           )
    target_sample = svc.target_sample
    # print(type(_audio), _audio.shape, target_sample)
    svc.clear_empty()

    # _audio = modify_speed(_audio, target_sample, speed=1.25)

    # output_file = "output.wav"
    # soundfile.write(output_file, _audio,
    #                 target_sample, format="wav")

    return target_sample, _audio
