import os
import traceback
from pathlib import Path
from pprint import pprint

import librosa
import numpy as np
import soundfile
import soundfile as sf

from inference.infer_tool import Svc
from . import tts_utils

enhance = False  # 是否使用NSF_HIFIGAN增强，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
shallow_diffusion = True  # 加载了扩散模型就启用浅扩散
only_diffusion = False  # "是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，不建议使用"
use_spk_mix = False  # 动态声线融合，需要手动编辑角色混合轨道，

auto_predict_f0 = True  # "自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）"
vc_transform = tran = 0  # 变调（整数，可以正负，半音数量，升高八度就是12

pad_seconds = 0.5  # 推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现
cl_num = clip_seconds = 0  # 音频自动切片，0为不切片，单位为秒(s)
lg_num = 0  # "两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s
lgr_num = 0.75  # 自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭
cluster_infer_ratio = 0.8  # 聚类模型混合比例，0-1之间，0即不启用聚类。使用聚类模型能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）
# F0_mean_pooling = False  # 是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭
f0_predictor = "crepe"  # 选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,默认为pm(注意：crepe为原F0使用均值滤波器)
cr_threshold = 0.05  # "F0过滤阈值，只有启动f0_mean_pooling时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音"

# 浅扩散
k_step = 100  # 浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果
second_encoding = False  # 二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭
loudness_envelope_adjustment = 0  # 输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络
# not suggest to modify
sid = "lain"
slice_db = -40  # 切片阈值
noice_scale = 0.4  # noise_scale 建议不要动，会影响音质，玄学参数

enhancer_adaptive_key = 0  # "使增强器适应更高的音域(单位为半音数)|默认为0"


class SvcInfer:
    @staticmethod
    def newFromSvc(svc: Svc):
        svcInfer = SvcInfer()
        svcInfer.model = svc
        return svcInfer

    def __init__(self, config=None):
        if config is not None:
            self.init_svc(config)

    def init_svc(self, config):
        device = "Auto"
        cluster_model_path = config["cluster_model_path"]
        fr = cluster_model_path.endswith(".pkl")  # 如果是pkl后缀就启用特征检索
        self.model = Svc(config["model_path"],
                         config["config_path"],
                         device=device if device != "Auto" else None,
                         cluster_model_path=cluster_model_path,
                         nsf_hifigan_enhance=enhance,
                         diffusion_model_path=config["diff_model_path"],
                         diffusion_config_path=config["diff_config_path"],
                         shallow_diffusion=shallow_diffusion,
                         only_diffusion=only_diffusion,
                         spk_mix_enable=use_spk_mix,
                         feature_retrieval=fr)

    def get_audio(self, text,
                  tts_engine="edge-tts",
                  language="ja",
                  speed=1.0,
                  **options) -> ((int, np.array), (int, np.array)):
        return self.tts_fn(text, tts_engine, language, speed,
                           "lain",
                           vc_transform, auto_predict_f0, cluster_infer_ratio,
                           slice_db, noice_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor,
                           enhancer_adaptive_key, cr_threshold,
                           k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, **options)

    def tts_fn(self, _text, tts_engine, _lang, _rate,
               sid, vc_transform, auto_f0, cluster_ratio,
               slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num,
               f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
               use_spk_mix, second_encoding, loudness_envelope_adjustment):
        # global model
        model = self.model
        # print("TTS")
        try:
            if model is None:
                return "你还没有加载模型", None
            if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
                if cluster_ratio != 0:
                    return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
            # _rate = f"+{int(_rate * 100)}%" if _rate >= 0 else f"{int(_rate * 100)}%"
            # _volume = f"+{int(_volume * 100)}%" if _volume >= 0 else f"{int(_volume * 100)}%"
            # cmd = [r"python", "tools/infer/tts.py", _text, _lang, _rate, _volume]
            target_sr = 44100
            origin_sampling_rate, origin_audio = tts_utils.text_to_audio(_text, tts_engine, _lang, _rate)
            # y, sr = librosa.load("tts.wav")
            # resampled_y = librosa.resample(origin_audio, orig_sr=origin_sampling_rate, target_sr=target_sr)
            # soundfile.write("tts.wav", resampled_y, target_sr, subtype="PCM_16")
            # input_audio = "tts.wav"
            # audio, sr = soundfile.read(input_audio)
            target_sampling_rate, target_audio = self.vc_infer("wav", sid, origin_audio, origin_sampling_rate,
                                                               vc_transform, auto_f0,
                                                               cluster_ratio,
                                                               slice_db, noise_scale, pad_seconds, cl_num, lg_num,
                                                               lgr_num, f0_predictor,
                                                               enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
                                                               second_encoding,
                                                               loudness_envelope_adjustment)
            # os.remove("tts.wav")
            # return "Success", output_file_path
            return (origin_sampling_rate, origin_audio), (target_sampling_rate, target_audio)

        except Exception as e:
            traceback.print_exc()
            raise e

    def vc_infer(self, output_format, sid, input_audio, sr, vc_transform, auto_f0, cluster_ratio,
                 slice_db,
                 noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold,
                 k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
        model = self.model
        if np.issubdtype(input_audio.dtype, np.integer):
            input_audio = (input_audio / np.iinfo(input_audio.dtype).max).astype(np.float32)
        if len(input_audio.shape) > 1:
            input_audio = librosa.to_mono(input_audio.transpose(1, 0))
        if sr != 44100:
            input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=44100)
        sf.write("temp.wav", input_audio, 44100, format="wav")

        pprint({
            "sid": sid,
            "vc_transform": vc_transform,
            "auto_f0": auto_f0,
            "cluster_ratio": cluster_ratio,
            "slice_db": slice_db,
            "noise_scale": noise_scale,
            "pad_seconds": pad_seconds,
            "cl_num": cl_num,
            "lg_num": lg_num,
            "lgr_num": lgr_num,
            "f0_predictor": f0_predictor,
            "enhancer_adaptive_key": enhancer_adaptive_key,
            "cr_threshold": cr_threshold,
            "k_step": k_step,
            "use_spk_mix": use_spk_mix,
            "second_encoding": second_encoding,
            "loudness_envelope_adjustment": loudness_envelope_adjustment
        })

        _audio = model.slice_inference(
            "temp.wav",
            sid,
            vc_transform,
            slice_db,
            cluster_ratio,
            auto_f0,
            noise_scale,
            pad_seconds,
            cl_num,
            lg_num,
            lgr_num,
            f0_predictor,
            enhancer_adaptive_key,
            cr_threshold,
            k_step,
            use_spk_mix,
            second_encoding,
            loudness_envelope_adjustment
        )
        model.clear_empty()
        if not os.path.exists("results"):
            os.makedirs("results")
        key = "auto" if auto_f0 else f"{int(vc_transform)}key"
        cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
        isdiffusion = "sovits"
        if model.shallow_diffusion:
            isdiffusion = "sovdiff"
        if model.only_diffusion:
            isdiffusion = "diff"
        # Gradio上传的filepath因为未知原因会有一个无意义的固定后缀，这里去掉
        # truncated_basename = Path(input_audio_path).stem[:-6] if Path(input_audio_path).stem[-6:] == "-0-100" else Path(
        #     input_audio_path).stem
        # output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
        # output_file_path = os.path.join("results", output_file_name)
        # if os.path.exists(output_file_path):
        #     count = 1
        #     while os.path.exists(output_file_path):
        #         output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}_{str(count)}.{output_format}'
        #         output_file_path = os.path.join("results", output_file_name)
        #         count += 1
        return model.target_sample, _audio
        # sf.write(output_file_path, _audio, model.target_sample, format=output_format)
        # return output_file_path
