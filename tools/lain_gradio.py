# ==================================================================================
# reference from
# https://www.yuque.com/umoubuton/ueupp5/sdahi7m5m6r0ur1r#zMbpu
# ==================================================================================
import argparse
import ast
import datetime
import glob
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from itertools import chain
from pathlib import Path

import gradio
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
import yaml
from gradio import networking

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools.infer.models_info import getModelsInfo
from tools.webui.webui_utils import Config, MODEL_TYPE, ENCODER_PRETRAIN
from tools.webui.release_packing import ReleasePacker
from tools.infer_base import SvcInfer
import utils
# from auto_slicer import AutoSlicer
from compress_model import removeOptimizer
from inference.infer_tool import Svc
from onnxexport.model_onnx import SynthesizerTrn
from utils import mix_model

root_project = Path(__file__).parent.parent.absolute()
os.chdir(root_project)

os.environ["PATH"] += os.pathsep + os.path.join(os.getcwd(), "ffmpeg", "bin")

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Some directories
workdir = "logs/44k"
second_dir = "models"
diff_second_dir = "models/diffusion"
diff_workdir = "logs/44k/diffusion"
config_dir = "configs/"
dataset_dir = "dataset/44k"
raw_path = "dataset_raw"
raw_wavs_path = "raw"
models_backup_path = 'models_backup'
root_dir = "checkpoints"
default_settings_file = "settings.yaml"
current_mode = ""
# Some global variables
debug = True
precheck_ok = False
model: Svc = None
svcInfer: SvcInfer = None
sovits_params = {}
diff_params = {}
# Some dicts for mapping


print("svcInfer", svcInfer)


def get_default_settings():
    global sovits_params, diff_params, second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    sovits_params = default_settings['sovits_params']
    diff_params = default_settings['diff_params']
    webui_settings = default_settings['webui_settings']
    second_dir_enable = webui_settings['second_dir']
    return sovits_params, diff_params, second_dir_enable


def webui_change(read_second_dir):
    global second_dir_enable
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    second_dir_enable = default_settings['webui_settings']['second_dir'] = read_second_dir
    config_file.save(default_settings)


def get_current_mode():
    global current_mode
    current_mode = "当前模式：独立目录模式，将从'./models/'读取模型文件" if second_dir_enable else "当前模式：工作目录模式，将从'./logs/44k'读取模型文件"
    return current_mode


def save_default_settings(log_interval, eval_interval, keep_ckpts, batch_size, learning_rate, amp_dtype, all_in_mem,
                          num_workers, cache_all_data, cache_device, diff_amp_dtype, diff_batch_size, diff_lr,
                          diff_interval_log, diff_interval_val, diff_force_save, diff_k_step_max):
    config_file = Config(default_settings_file, "yaml")
    default_settings = config_file.read()
    default_settings['sovits_params']['log_interval'] = int(log_interval)
    default_settings['sovits_params']['eval_interval'] = int(eval_interval)
    default_settings['sovits_params']['keep_ckpts'] = int(keep_ckpts)
    default_settings['sovits_params']['batch_size'] = int(batch_size)
    default_settings['sovits_params']['learning_rate'] = float(learning_rate)
    default_settings['sovits_params']['amp_dtype'] = str(amp_dtype)
    default_settings['sovits_params']['all_in_mem'] = all_in_mem
    default_settings['diff_params']['num_workers'] = int(num_workers)
    default_settings['diff_params']['cache_all_data'] = cache_all_data
    default_settings['diff_params']['cache_device'] = str(cache_device)
    default_settings['diff_params']['amp_dtype'] = str(diff_amp_dtype)
    default_settings['diff_params']['diff_batch_size'] = int(diff_batch_size)
    default_settings['diff_params']['diff_lr'] = float(diff_lr)
    default_settings['diff_params']['diff_interval_log'] = int(diff_interval_log)
    default_settings['diff_params']['diff_interval_val'] = int(diff_interval_val)
    default_settings['diff_params']['diff_force_save'] = int(diff_force_save)
    default_settings['diff_params']['diff_k_step_max'] = diff_k_step_max
    config_file.save(default_settings)
    return "成功保存默认配置"


def get_model_info(choice_ckpt):
    pthfile = os.path.join(ckpt_read_dir, choice_ckpt)
    net = torch.load(pthfile, map_location=torch.device('cpu'))  # cpu load to avoid using gpu memory
    spk_emb = net["model"].get("emb_g.weight")
    if spk_emb is None:
        return "所选模型缺少emb_g.weight，你可能选择了一个底模"
    _layer = spk_emb.size(1)
    encoder = [k for k, v in MODEL_TYPE.items() if v == _layer]  # 通过维度对应编码器
    encoder.sort()
    if encoder == ["hubertsoft", "vec256l9"]:
        encoder = ["vec256l9 / hubertsoft"]
    if encoder == ["cnhubertlarge", "whisper-ppg"]:
        encoder = ["whisper-ppg / cnhubertlarge"]
    if encoder == ["dphubert", "vec768l12", "wavlmbase+"]:
        encoder = ["vec768l12 / dphubert / wavlmbase+"]
    return encoder[0]


def load_json_encoder(config_choice, choice_ckpt):
    if config_choice == "no_config":
        return "未启用自动加载，请手动选择配置文件"
    if choice_ckpt == "no_model":
        return "请先选择模型"
    config_file = Config(os.path.join(config_read_dir, config_choice), "json")
    config = config_file.read()
    try:
        # 比对配置文件中的模型维度与该encoder的实际维度是否对应，防止古神语
        config_encoder = config["model"].get("speech_encoder", "no_encoder")
        config_dim = config["model"]["ssl_dim"]
        # 旧版配置文件自动匹配
        if config_encoder == "no_encoder":
            config_encoder = config["model"]["speech_encoder"] = "vec256l9" if config_dim == 256 else "vec768l12"
            config_file.save(config)
        correct_dim = MODEL_TYPE.get(config_encoder, "unknown")
        if config_dim != correct_dim:
            return "配置文件中的编码器与模型维度不匹配"
        return config_encoder
    except Exception as e:
        return f"出错了: {e}"


def auto_load(choice_ckpt):
    global second_dir_enable
    model_output_msg = get_model_info(choice_ckpt)
    json_output_msg = config_choice = ""
    choice_ckpt_name, _ = os.path.splitext(choice_ckpt)
    if second_dir_enable:
        all_config = [json for json in os.listdir(second_dir) if json.endswith(".json")]
        for config in all_config:
            config_fname, _ = os.path.splitext(config)
            if config_fname == choice_ckpt_name:
                config_choice = config
                json_output_msg = load_json_encoder(config, choice_ckpt)
        if json_output_msg != "":
            return model_output_msg, config_choice, json_output_msg
        else:
            return model_output_msg, "no_config", ""
    else:
        return model_output_msg, "no_config", ""


def auto_load_diff(diff_model):
    global second_dir_enable
    if second_dir_enable is False:
        return "no_diff_config"
    all_diff_config = [yaml for yaml in os.listdir(second_dir) if yaml.endswith(".yaml")]
    for config in all_diff_config:
        config_fname, _ = os.path.splitext(config)
        diff_fname, _ = os.path.splitext(diff_model)
        if config_fname == diff_fname:
            return config
    return "no_diff_config"


def load_model_func(ckpt_name, cluster_name, config_name, enhance, diff_model_name, diff_config_name, only_diffusion,
                    use_spk_mix, using_device, method, speedup, cl_num):
    global model
    config_path = os.path.join(config_read_dir, config_name) if not only_diffusion else "configs/config.json"
    diff_config_path = os.path.join(config_read_dir,
                                    diff_config_name) if diff_config_name != "no_diff_config" else "configs/diffusion.yaml"
    ckpt_path = os.path.join(ckpt_read_dir, ckpt_name)
    cluster_path = os.path.join(ckpt_read_dir, cluster_name)
    diff_model_path = os.path.join(diff_read_dir, diff_model_name)
    k_step_max = 1000
    if not only_diffusion:
        config = Config(config_path, "json").read()
    if diff_model_name != "no_diff":
        _diff = Config(diff_config_path, "yaml")
        _content = _diff.read()
        diff_spk = _content.get('spk', {})
        diff_spk_choice = spk_choice = next(iter(diff_spk), "未检测到音色")
        if not only_diffusion:
            if _content['data'].get('encoder_out_channels') != config["model"].get('ssl_dim'):
                return "扩散模型维度与主模型不匹配，请确保两个模型使用的是同一个编码器", gr.Dropdown.update(choices=[],
                                                                                                           value=""), 0, None
        _content["infer"]["speedup"] = int(speedup)
        _content["infer"]["method"] = str(method)
        k_step_max = _content["model"].get('k_step_max', 0) if _content["model"].get('k_step_max', 0) != 0 else 1000
        _diff.save(_content)
    encoder = None
    if not only_diffusion:
        net = torch.load(ckpt_path, map_location=torch.device('cpu'))
        # 读取模型各维度并比对，还有小可爱无视提示硬要加载底模的就返回个未初始张量
        emb_dim, model_dim = net["model"].get("emb_g.weight", torch.empty(0, 0)).size()
        if emb_dim > config["model"]["n_speakers"]:
            return "模型说话人数量与emb维度不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
        if model_dim != config["model"]["ssl_dim"]:
            return "配置文件与模型不匹配", gr.Dropdown.update(choices=[], value=""), 0, None
        encoder = config["model"]["speech_encoder"]
        spk_dict = config.get('spk', {})
        spk_choice = next(iter(spk_dict), "未检测到音色")
    else:
        spk_dict = diff_spk
        spk_choice = diff_spk_choice

    fr = cluster_name.endswith(".pkl")  # 如果是pkl后缀就启用特征检索
    shallow_diffusion = diff_model_name != "no_diff"  # 加载了扩散模型就启用浅扩散
    device = cuda[using_device] if "CUDA" in using_device else using_device
    if model is not None and \
            model.nsf_hifigan_enhance == enhance and \
            model.shallow_diffusion == shallow_diffusion and \
            model.only_diffusion == only_diffusion and \
            model.spk_mix_enable == use_spk_mix and \
            model.feature_retrieval == fr:
        print("模型已加载，无需重复加载")
    else:
        model_empty_cache()
        model = Svc(ckpt_path,
                    config_path,
                    device=device if device != "Auto" else None,
                    cluster_model_path=cluster_path,
                    nsf_hifigan_enhance=enhance,
                    diffusion_model_path=diff_model_path,
                    diffusion_config_path=diff_config_path,
                    shallow_diffusion=shallow_diffusion,
                    only_diffusion=only_diffusion,
                    spk_mix_enable=use_spk_mix,
                    feature_retrieval=fr)
        global svcInfer
        svcInfer = SvcInfer.newFromSvc(model)
    spk_list = list(spk_dict.keys())
    if not only_diffusion:
        clip = 25 if encoder == "whisper-ppg" or encoder == "whisper-ppg-large" else cl_num  # Whisper必须强制切片25秒
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        sovits_msg = f"模型被成功加载到了{device_name}上\n"
    else:
        clip = cl_num
        sovits_msg = "启用全扩散推理，未加载So-VITS模型\n"
    index_or_kmeans = "特征索引" if fr else "聚类模型"
    clu_load = "未加载" if cluster_name == "no_clu" else cluster_name
    diff_load = "未加载" if diff_model_name == "no_diff" else f"{diff_model_name} | 采样器: {method} | 加速倍数：{int(speedup)} | 最大浅扩散步数：{k_step_max}"
    output_msg = f"{sovits_msg}{index_or_kmeans}：{clu_load}\n扩散模型：{diff_load}"

    return (
        output_msg,
        gr.Dropdown.update(choices=spk_list, value=spk_choice),
        clip,
        gr.Slider.update(value=100 if k_step_max > 100 else k_step_max, minimum=speedup, maximum=k_step_max),

        auto_load(choice_ckpt.value),
        load_json_encoder(config_choice.value, choice_ckpt.value),
    )


def model_empty_cache():
    global model
    if model is None:
        return sid.update(choices=[], value=""), "没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices=[], value=""), "模型卸载完毕!"


def get_file_options(directory, extension):
    return [file for file in os.listdir(directory) if file.endswith(extension)]


def load_options():
    ckpt_list = [file for file in get_file_options(ckpt_read_dir, ".pth") if
                 not file.startswith("D_") or file == "G_0.pth"]
    config_list = get_file_options(config_read_dir, ".json")
    cluster_list = ["no_clu"] + get_file_options(ckpt_read_dir, ".pt") + get_file_options(ckpt_read_dir,
                                                                                          ".pkl")  # 聚类和特征检索模型
    diff_list = ["no_diff"] + get_file_options(diff_read_dir, ".pt")
    diff_config_list = ["no_diff_config"] + get_file_options(config_read_dir, ".yaml")
    return ckpt_list, config_list, cluster_list, diff_list, diff_config_list


def refresh_options():
    global ckpt_read_dir, config_read_dir, diff_read_dir, current_mode
    ckpt_read_dir = second_dir if second_dir_enable else workdir
    config_read_dir = second_dir if second_dir_enable else config_dir
    diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
    ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()
    current_mode = get_current_mode()
    return (
        choice_ckpt.update(choices=ckpt_list),
        config_choice.update(choices=config_list),
        cluster_choice.update(choices=cluster_list),
        diff_choice.update(choices=diff_list),
        diff_config_choice.update(choices=diff_config_list),
        mode_caption.update(value=f"""{current_mode}，可在页面底端切换模式""")
    )


def source_change(use_microphone):
    if use_microphone:
        return vc_input3.update(source="microphone")
    else:
        return vc_input3.update(source="upload")


def vc_infer(output_format, sid, input_audio, sr, input_audio_path, vc_transform, auto_f0, cluster_ratio, slice_db,
             noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold,
             k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    if np.issubdtype(input_audio.dtype, np.integer):
        input_audio = (input_audio / np.iinfo(input_audio.dtype).max).astype(np.float32)
    if len(input_audio.shape) > 1:
        input_audio = librosa.to_mono(input_audio.transpose(1, 0))
    if sr != 44100:
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=44100)
    sf.write("temp.wav", input_audio, 44100, format="wav")
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
    truncated_basename = Path(input_audio_path).stem[:-6] if Path(input_audio_path).stem[-6:] == "-0-100" else Path(
        input_audio_path).stem
    output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file_path = os.path.join("results", output_file_name)
    if os.path.exists(output_file_path):
        count = 1
        while os.path.exists(output_file_path):
            output_file_name = f'{truncated_basename}_{sid}_{key}{cluster}{isdiffusion}_{str(count)}.{output_format}'
            output_file_path = os.path.join("results", output_file_name)
            count += 1
    sf.write(output_file_path, _audio, model.target_sample, format=output_format)
    return output_file_path


def vc_fn(output_format, sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
          cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
          second_encoding, loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "你还没有上传音频", None
        if model is None:
            return "你还没有加载模型", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
        audio, sr = sf.read(input_audio)
        output_file_path = vc_infer(output_format, sid, audio, sr, input_audio, vc_transform, auto_f0, cluster_ratio,
                                    slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor,
                                    enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding,
                                    loudness_envelope_adjustment)
        os.remove("temp.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def vc_batch_fn(output_format, sid, input_audio_files, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
                pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
                use_spk_mix, second_encoding, loudness_envelope_adjustment, progress=gr.Progress()):
    global model
    try:
        if input_audio_files is None or len(input_audio_files) == 0:
            return "你还没有上传音频"
        if model is None:
            return "你还没有加载模型"
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "你还未加载聚类或特征检索模型，无法启用聚类/特征检索混合比例", None
        _output = []
        for file_obj in progress.tqdm(input_audio_files, desc="Inferencing"):
            print(f"Start processing: {file_obj.name}")
            input_audio_path = file_obj.name
            audio, sr = sf.read(input_audio_path)
            output_file_path = vc_infer(output_format, sid, sr, audio, input_audio_path, vc_transform, auto_f0,
                                        cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num,
                                        f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
                                        second_encoding, loudness_envelope_adjustment)
            _output.append(output_file_path)
        return "批量推理完成，音频已经被保存到results文件夹"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def tts_fn(_text, tts_engine, _lang, _rate, output_format, sid, vc_transform, auto_f0, cluster_ratio, slice_db,
           noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step,
           use_spk_mix, second_encoding, loudness_envelope_adjustment):
    try:
        res = svcInfer.get_audio(_text, tts_engine, _lang, _rate,
                                 spk=sid,
                                 tran=vc_transform,
                                 slice_db=slice_db,
                                 cluster_ratio=cluster_ratio,
                                 auto_predict_f0=auto_f0,
                                 # auto_f0, cluster_ratio,
                                 noise_scale=noise_scale,
                                 pad_seconds=pad_seconds,
                                 clip_seconds=cl_num,
                                 lg_num=lg_num,
                                 lgr_num=lgr_num,
                                 # cl_num, lg_num, lgr_num,
                                 f0_predictor=f0_predictor,
                                 enhancer_adaptive_key=enhancer_adaptive_key,
                                 cr_threshold=cr_threshold, k_step=k_step,
                                 use_spk_mix=use_spk_mix,
                                 second_encoding=second_encoding,
                                 loudness_envelope_adjustment=loudness_envelope_adjustment)
        # os.remove("tts.wav")
        return res
        sample, _audio = res[1]

        key = "auto" if auto_f0 else f"{int(vc_transform)}key"
        cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
        isdiffusion = "sovits"
        if model.shallow_diffusion:
            isdiffusion = "sovdiff"
        if model.only_diffusion:
            isdiffusion = "diff"
        tmp_file_path = tempfile.mkstemp()[1]
        output_file_path = f'{tmp_file_path}_{sid}_{key}{cluster}{isdiffusion}.{output_format}'

        sf.write(output_file_path, _audio, sample, format=output_format)
        return "Success", output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def load_raw_dirs():
    global precheck_ok
    precheck_ok = False
    allowed_pattern = re.compile(r'^[a-zA-Z0-9_@#$%^&()_+\-=\s\.]*$')
    illegal_files = illegal_dataset = []
    for root, dirs, files in os.walk(raw_path):
        for dir in dirs:
            if not allowed_pattern.match(dir):
                illegal_dataset.append(dir)
        if illegal_dataset:
            return f"数据集文件夹名只能包含数字、字母、下划线，以下文件夹不符合要求，请改名后再试：\n{illegal_dataset}"
        if root != raw_path:  # 只处理子文件夹内的文件
            for file in files:
                if not allowed_pattern.match(file) and file not in illegal_files:
                    illegal_files.append(file)
                if not file.lower().endswith('.wav') and file not in illegal_files:
                    illegal_files.append(file)
    if illegal_files:
        return f"数据集文件名只能包含数字、字母、下划线，且必须是.wav格式，以下文件不符合要求，请改名后再试：\n{illegal_files}"
    spk_dirs = [entry.name for entry in os.scandir(raw_path) if entry.is_dir()]
    if spk_dirs:
        precheck_ok = True
        return spk_dirs
    else:
        return "未找到数据集，请检查dataset_raw文件夹"


def dataset_preprocess(encoder, f0_predictor, use_diff, vol_aug, skip_loudnorm, num_processes):
    if precheck_ok:
        diff_arg = "--use_diff" if use_diff else ""
        vol_aug_arg = "--vol_aug" if vol_aug else ""
        skip_loudnorm_arg = "--skip_loudnorm" if skip_loudnorm else ""
        preprocess_commands = [
            r".\workenv\python.exe resample.py %s" % (skip_loudnorm_arg),
            r".\workenv\python.exe preprocess_flist_config.py --speech_encoder %s %s" % (encoder, vol_aug_arg),
            r".\workenv\python.exe preprocess_hubert_f0.py --num_processes %s --f0_predictor %s %s" % (
                num_processes, f0_predictor, diff_arg)
        ]
        accumulated_output = ""
        # 清空dataset
        dataset = os.listdir(dataset_dir)
        if len(dataset) != 0:
            for dir in dataset:
                dataset_spk_dir = os.path.join(dataset_dir, str(dir))
                if os.path.isdir(dataset_spk_dir):
                    shutil.rmtree(dataset_spk_dir)
                    accumulated_output += f"Deleting previous dataset: {dir}\n"
        for command in preprocess_commands:
            try:
                result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                          text=True)
                accumulated_output += f"Command: {command}, Using Encoder: {encoder}, Using f0 Predictor: {f0_predictor}\n"
                yield accumulated_output, None
                progress_line = None
                for line in result.stdout:
                    if r"it/s" in line or r"s/it" in line:  # 防止进度条刷屏
                        progress_line = line
                    else:
                        accumulated_output += line
                    if progress_line is None:
                        yield accumulated_output, None
                    else:
                        yield accumulated_output + progress_line, None
                result.communicate()
            except subprocess.CalledProcessError as e:
                result = e.output
                accumulated_output += f"Error: {result}\n"
                yield accumulated_output, None
            if progress_line is not None:
                accumulated_output += progress_line
            accumulated_output += '-' * 50 + '\n'
            yield accumulated_output, None
            config_path = "configs/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        spk_name = config.get('spk', None)
        yield accumulated_output, gr.Textbox.update(value=spk_name)
    else:
        yield "数据集识别未通过，请先识别数据集并确保没有报错信息", None


def regenerate_config(encoder, vol_aug):
    if precheck_ok is False:
        return "数据集识别未通过，请检查识别结果的报错信息"
    vol_aug_arg = "--vol_aug" if vol_aug else ""
    cmd = r".\workenv\python.exe preprocess_flist_config.py --speech_encoder %s %s" % (encoder, vol_aug_arg)
    output = ""
    try:
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
        for line in result.stdout:
            output += line
        output += "Regenerate config file successfully."
    except subprocess.CalledProcessError as e:
        result = e.output
        output += f"Error: {result}\n"
    return output


def clear_output():
    return gr.Textbox.update(value="Cleared!>_<")


def get_available_encoder():
    current_pretrain = os.listdir("pretrain")
    current_pretrain = [("pretrain/" + model) for model in current_pretrain]
    encoder_list = []
    for encoder, path in ENCODER_PRETRAIN.items():
        if path in current_pretrain:
            encoder_list.append(encoder)
    return encoder_list


def config_fn(log_interval, eval_interval, keep_ckpts, batch_size, lr, amp_dtype, all_in_mem, diff_num_workers,
              diff_cache_all_data, diff_batch_size, diff_lr, diff_interval_log, diff_interval_val, diff_cache_device,
              diff_amp_dtype, diff_force_save, diff_k_step_max):
    if amp_dtype == "fp16" or amp_dtype == "bf16":
        fp16_run = True
    else:
        fp16_run = False
        amp_dtype = "fp16"
    config_origin = Config("configs/config.json", "json")
    diff_config = Config("configs/diffusion.yaml", "yaml")
    config_data = config_origin.read()
    config_data['train']['log_interval'] = int(log_interval)
    config_data['train']['eval_interval'] = int(eval_interval)
    config_data['train']['keep_ckpts'] = int(keep_ckpts)
    config_data['train']['batch_size'] = int(batch_size)
    config_data['train']['learning_rate'] = float(lr)
    config_data['train']['fp16_run'] = fp16_run
    config_data['train']['half_type'] = str(amp_dtype)
    config_data['train']['all_in_mem'] = all_in_mem
    config_origin.save(config_data)
    diff_config_data = diff_config.read()
    diff_config_data['train']['num_workers'] = int(diff_num_workers)
    diff_config_data['train']['cache_all_data'] = diff_cache_all_data
    diff_config_data['train']['batch_size'] = int(diff_batch_size)
    diff_config_data['train']['lr'] = float(diff_lr)
    diff_config_data['train']['interval_log'] = int(diff_interval_log)
    diff_config_data['train']['interval_val'] = int(diff_interval_val)
    diff_config_data['train']['cache_device'] = str(diff_cache_device)
    diff_config_data['train']['amp_dtype'] = str(diff_amp_dtype)
    diff_config_data['train']['interval_force_save'] = int(diff_force_save)
    diff_config_data['model']['k_step_max'] = 100 if diff_k_step_max else 0
    diff_config.save(diff_config_data)
    return "配置文件写入完成"


def check_dataset(dataset_path):
    if not os.listdir(dataset_path):
        return "数据集不存在，请检查dataset文件夹"
    no_npy_pt_files = True
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.pt'):
                no_npy_pt_files = False
                break
    if no_npy_pt_files:
        return "数据集中未检测到f0和hubert文件，可能是预处理未完成"
    return None


def training(gpu_selection, encoder):
    config_file = Config("configs/config.json", "json")
    config_data = config_file.read()
    vol_emb = config_data["model"]["vol_embedding"]
    dataset_warn = check_dataset(dataset_dir)
    if dataset_warn is not None:
        return dataset_warn
    PRETRAIN = {
        "vec256l9": ("D_0.pth", "G_0.pth", "pre_trained_model"),
        "vec768l12": (
            "D_0.pth", "G_0.pth", "pre_trained_model/768l12/vol_emb" if vol_emb else "pre_trained_model/768l12"),
        "hubertsoft": ("D_0.pth", "G_0.pth", "pre_trained_model/hubertsoft"),
        "whisper-ppg": ("D_0.pth", "G_0.pth", "pre_trained_model/whisper-ppg"),
        "cnhubertlarge": ("D_0.pth", "G_0.pth", "pre_trained_model/cnhubertlarge"),
        "dphubert": ("D_0.pth", "G_0.pth", "pre_trained_model/dphubert"),
        "wavlmbase+": ("D_0.pth", "G_0.pth", "pre_trained_model/wavlmbase+"),
        "whisper-ppg-large": ("D_0.pth", "G_0.pth", "pre_trained_model/whisper-ppg-large")
    }
    if encoder not in PRETRAIN:
        return "未知编码器"
    d_0_file, g_0_file, encoder_model_path = PRETRAIN[encoder]
    d_0_path = os.path.join(encoder_model_path, d_0_file)
    g_0_path = os.path.join(encoder_model_path, g_0_file)
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    new_backup_folder = os.path.join(models_backup_path, str(timestamp))
    output_msg = ""
    if os.listdir(workdir) != ['diffusion']:
        os.makedirs(new_backup_folder, exist_ok=True)
        for file in os.listdir(workdir):
            if file != "diffusion":
                shutil.move(os.path.join(workdir, file), os.path.join(new_backup_folder, file))
    if os.path.isfile(g_0_path) and os.path.isfile(d_0_path):
        shutil.copy(d_0_path, os.path.join(workdir, "D_0.pth"))
        shutil.copy(g_0_path, os.path.join(workdir, "G_0.pth"))
        output_msg += f"成功装载预训练模型，编码器：{encoder}\n"
    else:
        output_msg += f"{encoder}的预训练模型不存在，未装载预训练模型\n"
    cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (
        gpu_selection)
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    output_msg += "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"
    return output_msg


def continue_training(gpu_selection, encoder):
    dataset_warn = check_dataset(dataset_dir)
    if dataset_warn is not None:
        return dataset_warn
    if encoder == "":
        return "请先选择预处理对应的编码器"
    all_files = os.listdir(workdir)
    model_files = [f for f in all_files if f.startswith('G_') and f.endswith('.pth')]
    if len(model_files) == 0:
        return "你还没有已开始的训练"
    cmd = r"set CUDA_VISIBLE_DEVICES=%s && .\workenv\python.exe train.py -c configs/config.json -m 44k" % (
        gpu_selection)
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"


def kmeans_training(kmeans_gpu):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    cmd = r".\workenv\python.exe cluster/train_cluster.py --gpu" if kmeans_gpu else r".\workenv\python.exe cluster/train_cluster.py"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练，训练聚类模型不会输出日志，CPU训练一般需要5-10分钟左右"


def index_training():
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    cmd = r".\workenv\python.exe train_index.py -c configs/config.json"
    subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    return "已经在新的终端窗口开始训练"


def diff_training(encoder, k_step_max):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    new_backup_folder = os.path.join(models_backup_path, "diffusion", str(timestamp))
    if len(os.listdir(diff_workdir)) != 0:
        os.makedirs(new_backup_folder, exist_ok=True)
        for file in os.listdir(diff_workdir):
            shutil.move(os.path.join(diff_workdir, file), os.path.join(new_backup_folder, file))
    DIFF_PRETRAIN = {
        "768-kstepmax100": "pre_trained_model/diffusion/768l12/max100/model_0.pt",
        "vec768l12": "pre_trained_model/diffusion/768l12/model_0.pt",
        "hubertsoft": "pre_trained_model/diffusion/hubertsoft/model_0.pt",
        "whisper-ppg": "pre_trained_model/diffusion/whisper-ppg/model_0.pt"
    }
    if encoder not in DIFF_PRETRAIN:
        return "你所选的编码器暂时不支持训练扩散模型"
    if k_step_max:
        encoder = "768-kstepmax100"
    diff_pretrained_model = DIFF_PRETRAIN[encoder]
    shutil.copy(diff_pretrained_model, os.path.join(diff_workdir, "model_0.pt"))
    subprocess.Popen(
        ["cmd", "/c", "start", "cmd", "/k", r".\workenv\python.exe train_diff.py -c configs/diffusion.yaml"])
    output_message = "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"
    if encoder == "768-kstepmax100":
        output_message += "\n正在进行100步深度的浅扩散训练，已加载底模"
    else:
        output_message += f"\n正在进行完整深度的扩散训练，编码器{encoder}"
    return output_message


def diff_continue_training(encoder):
    if not os.listdir(dataset_dir):
        return "数据集不存在，请检查dataset文件夹"
    if encoder == "":
        return "请先选择预处理对应的编码器"
    all_files = os.listdir(diff_workdir)
    model_files = [f for f in all_files if f.endswith('.pt')]
    if len(model_files) == 0:
        return "你还没有已开始的训练"
    subprocess.Popen(
        ["cmd", "/c", "start", "cmd", "/k", r".\workenv\python.exe train_diff.py -c configs/diffusion.yaml"])
    return "已经在新的终端窗口开始训练，请监看终端窗口的训练日志。在终端中按Ctrl+C可暂停训练。"


def upload_mix_append_file(files, sfiles):
    try:
        if (sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files, sfiles)]
        p = {file: 100 for file in file_paths}
        return file_paths, mix_model_output1.update(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def mix_submit_click(js, mode):
    try:
        assert js.lstrip() != ""
        modes = {"凸组合": 0, "线性组合": 1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path, mix_rate = zip(*data)
        path = mix_model(model_path, mix_rate, mode)
        return f"成功，文件被保存在了{path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def updata_mix_info(files):
    try:
        if files is None:
            return mix_model_output1.update(value="")
        p = {file.name: 100 for file in files}
        return mix_model_output1.update(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def pth_identify():
    if not os.path.exists(root_dir):
        return f"未找到{root_dir}文件夹，请先创建一个{root_dir}文件夹并按第一步流程操作"
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not model_dirs:
        return f"未在{root_dir}文件夹中找到模型文件夹，请确保每个模型和配置文件都被放置在单独的文件夹中"
    valid_model_dirs = []
    for path in model_dirs:
        pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
        json_files = glob.glob(f"{root_dir}/{path}/*.json")
        if len(pth_files) != 1 or len(json_files) != 1:
            return f"错误: 在{root_dir}/{path}中找到了{len(pth_files)}个.pth文件和{len(json_files)}个.json文件。应当确保每个文件夹内有且只有一个.pth文件和.json文件"
        valid_model_dirs.append(path)

    return f"成功识别了{len(valid_model_dirs)}个模型：{valid_model_dirs}"


def onnx_export():
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    try:
        for path in model_dirs:
            pth_files = glob.glob(f"{root_dir}/{path}/*.pth")
            json_files = glob.glob(f"{root_dir}/{path}/*.json")
            model_file = pth_files[0]
            json_file = json_files[0]
            device = torch.device("cpu")
            hps = utils.get_hparams_from_file(json_file)
            SVCVITS = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model)
            _ = utils.load_checkpoint(model_file, SVCVITS, None)
            _ = SVCVITS.eval().to(device)
            for i in SVCVITS.parameters():
                i.requires_grad = False
            n_frame = 10
            test_hidden_unit = torch.rand(1, n_frame, 256)
            test_pitch = torch.rand(1, n_frame)
            test_mel2ph = torch.arange(0, n_frame, dtype=torch.int64)[
                None]  # torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
            test_uv = torch.ones(1, n_frame, dtype=torch.float32)
            test_noise = torch.randn(1, 192, n_frame)
            test_sid = torch.LongTensor([0])
            input_names = ["c", "f0", "mel2ph", "uv", "noise", "sid"]
            output_names = ["audio", ]
            onnx_file = os.path.splitext(model_file)[0] + ".onnx"
            torch.onnx.export(SVCVITS,
                              (
                                  test_hidden_unit.to(device),
                                  test_pitch.to(device),
                                  test_mel2ph.to(device),
                                  test_uv.to(device),
                                  test_noise.to(device),
                                  test_sid.to(device)
                              ),
                              onnx_file,
                              dynamic_axes={
                                  "c": [0, 1],
                                  "f0": [1],
                                  "mel2ph": [1],
                                  "uv": [1],
                                  "noise": [2],
                              },
                              do_constant_folding=False,
                              opset_version=16,
                              verbose=False,
                              input_names=input_names,
                              output_names=output_names)
        return "转换成功，模型被保存在了checkpoints下的对应目录"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def load_raw_audio(audio_path):
    if not os.path.isdir(audio_path):
        return "请输入正确的目录", None
    files = os.listdir(audio_path)
    wav_files = [file for file in files if file.lower().endswith('.wav')]
    if not wav_files:
        return "未在目录中找到.wav音频文件", None
    return "成功加载", wav_files


def slicer_fn(input_dir, output_dir, process_method, max_sec, min_sec):
    if output_dir == "":
        return "请先选择输出的文件夹"
    if output_dir == input_dir:
        return "输出目录不能和输入目录相同"
    slicer = AutoSlicer()
    if os.path.exists(output_dir) is not True:
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            slicer.auto_slice(filename, input_dir, output_dir, max_sec)
    if process_method == "丢弃":
        for filename in os.listdir(output_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(output_dir, filename)
                audio, sr = librosa.load(filepath, sr=None, mono=False)
                if librosa.get_duration(y=audio, sr=sr) < min_sec:
                    os.remove(filepath)
    elif process_method == "将过短音频整合为长音频":
        slicer.merge_short(output_dir, max_sec, min_sec)
    file_count, max_duration, min_duration, orig_duration, final_duration = slicer.slice_count(input_dir, output_dir)
    hrs = int(final_duration / 3600)
    mins = int((final_duration % 3600) / 60)
    sec = format(float(final_duration % 60), '.2f')
    rate = format(100 * (final_duration / orig_duration), '.2f') if orig_duration != 0 else 0
    rate_msg = f"为原始音频时长的{rate}%" if rate != 0 else "因未知问题，无法计算切片时长的占比"
    return f"成功将音频切分为{file_count}条片段，其中最长{max_duration}秒，最短{min_duration}秒，切片后的音频总时长{hrs:02d}小时{mins:02d}分{sec}秒，{rate_msg}"


def model_compression(_model):
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        model_path = os.path.join(ckpt_read_dir, _model)
        filename, extension = os.path.splitext(_model)
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(ckpt_read_dir, output_model_name)
        removeOptimizer(model_path, output_path)
        return f"模型已成功被保存在了{output_path}"


def pack_autoload(model_to_pack):
    _, config_name, _ = auto_load(model_to_pack)
    if config_name == "no_config":
        return "未找到对应的配置文件，请手动选择", None
    else:
        _config = Config(os.path.join(config_read_dir, config_name), "json")
        _content = _config.read()
        spk_dict = _content["spk"]
        spk_list = ",".join(spk_dict.keys())
        return config_name, spk_list


def release_packing(model_to_pack, model_config, speaker, diff_to_pack, cluster_to_pack):
    print("release_packing")
    print(model_to_pack, model_config, speaker, diff_to_pack, cluster_to_pack)
    model_path = diff_path = cluster_path = ""
    basename = os.path.splitext(model_to_pack)[0]
    diff_basename = os.path.splitext(diff_to_pack)[0]
    if model_to_pack == "" or model_config == "" or speaker == "":
        return "存在必选项为空，请检查后重试"
    released_pack = ReleasePacker(speaker, model_to_pack)
    released_pack.remove_temp("release_packs")
    model_path = os.path.join(ckpt_read_dir, model_to_pack)
    if os.stat(model_path).st_size > 300000000:
        removeOptimizer(model_path, os.path.join("release_packs", model_to_pack))
        model_path = os.path.join("release_packs", model_to_pack)
    if diff_to_pack != "no_diff":
        diff_path = os.path.join(diff_read_dir, diff_to_pack)
    if cluster_to_pack != "no_cluster":
        cluster_path = os.path.join(ckpt_read_dir, cluster_to_pack)
    shutil.copyfile("configs_template/config_template.json", "release_packs/config_template.json")
    shutil.copyfile("configs_template/diffusion_template.yaml", "release_packs/diffusion_template.yaml")
    files_to_pack = [
        (model_path, f"models/{model_to_pack}"),
        (diff_path, f"models/diffusion/{diff_to_pack}") if diff_to_pack != "no_diff" else ("", ""),
        (cluster_path, f"models/{cluster_to_pack}") if cluster_to_pack != "no_cluster" else ("", ""),
        (f"release_packs/{basename}.json", f"models/{basename}.json"),
        (f"release_packs/{diff_basename}.yaml", f"models/{diff_basename}.yaml") if diff_to_pack != "no_diff" else (
            "", ""),
        ("release_packs/install.txt", "install.txt")
    ]
    released_pack.add_file(files_to_pack)
    released_pack.generate_config(diff_to_pack, model_config)
    os.rename("release_packs/config_template.json", f"release_packs/{basename}.json")
    os.rename("release_packs/diffusion_template.yaml", f"release_packs/{diff_basename}.yaml")
    released_pack.pack()
    to_remove = [file for file in os.listdir("release_packs") if not file.endswith(".zip")]
    for file in to_remove:
        os.remove(os.path.join("release_packs", file))
    return "打包成功, 请在release_packs目录下查看"


def release_install(model_zip_path):
    model_zip = ReleasePacker("", "")
    model_zip.unpack(model_zip_path)
    for file in os.listdir("release_packs"):
        if file.endswith(".txt"):
            install_txt = os.path.join("release_packs", file)
            break
    else:
        model_zip.remove_temp("release_packs")
        return "非格式化安装包，无法安装"
    _spk = model_zip.formatted_install(install_txt)
    model_zip.remove_temp("release_packs")
    return f"安装成功，可用说话人{_spk}，请启用独立目录模式加载模型"


# read default params
sovits_params, diff_params, second_dir_enable = get_default_settings()
second_dir_enable = True
second_dir = root_project  # / "models"
diff_second_dir = second_dir  # / "diffusion"
ckpt_read_dir = second_dir if second_dir_enable else workdir
config_read_dir = second_dir if second_dir_enable else config_dir
diff_read_dir = diff_second_dir if second_dir_enable else diff_workdir
current_mode = get_current_mode()

# create dirs if they don't exist
dirs_to_check = [
    workdir,
    second_dir,
    diff_workdir,
    diff_second_dir,
    dataset_dir,
]
for dir in dirs_to_check:
    if not os.path.exists(dir):
        os.makedirs(dir)

# read ckpt list
ckpt_list, config_list, cluster_list, diff_list, diff_config_list = load_options()

# read available encoder list
encoder_list = get_available_encoder()

# read GPU info
ngpu = torch.cuda.device_count()
gpu_infos = []
if (torch.cuda.is_available() is False or ngpu == 0):
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if ("MX" in gpu_name):
            continue
        if (
                "RTX" in gpu_name.upper() or "10" in gpu_name or "16" in gpu_name or "20" in gpu_name or "30" in gpu_name or "40" in gpu_name or "A50" in gpu_name.upper() or "70" in gpu_name or "80" in gpu_name or "90" in gpu_name or "M4" in gpu_name or "P4" in gpu_name or "T4" in gpu_name or "TITAN" in gpu_name.upper()):  # A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
gpu_info = "\n".join(gpu_infos) if if_gpu_ok is True and len(gpu_infos) > 0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus = "-".join([i[0] for i in gpu_infos])

# read cuda info for inference
cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

# Check BF16 support
amp_options = ["fp32", "fp16"]
if if_gpu_ok:
    if torch.cuda.is_bf16_supported():
        amp_options = ["fp32", "fp16", "bf16"]

# Get F0 Options
f0_options = ["crepe", "pm", "dio", "harvest", "rmvpe"] if os.path.exists("pretrain/rmvpe.pt") else ["crepe", "pm",
                                                                                                     "dio", "harvest"]

print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model info name')
parser.add_argument('--model_path', type=str, help='model_path')
parser.add_argument('--config_path', type=str, help='config_path')
parser.add_argument('--diff_model_path', type=str, help='diff_model_path')
parser.add_argument('--diff_config_path', type=str, help='diff_config_path')
parser.add_argument('--cluster_model_path', type=str, help='cluster_model_path')
parser.add_argument("--port", type=int, help='port')
parser.add_argument("--root_path", type=str, help='root_path', default="")
parser.add_argument("--share", action='store_true', help='share')
parser.add_argument('--debug', action='store_true', help='debug')
# parser.add_argument('--inbrowser', action='store_true', help='inbrowser')
args = parser.parse_args()

if args.model is not None:
    modelInfo = getModelsInfo(args.model)
    config = modelInfo.to_svc_config(str(root_project / "models"))
    default_model_path = config["model_path"]
    default_config_path = config["config_path"]
    default_diff_model_path = config["diff_model_path"]
    default_diff_config_path = config["diff_config_path"]
    default_cluster_model_path = config["cluster_model_path"]
else:
    default_model_path = args.model_path
    default_config_path = args.config_path
    default_diff_model_path = args.diff_model_path
    default_diff_config_path = args.diff_config_path
    default_cluster_model_path = args.cluster_model_path

app = gr.Blocks()
with app:
    gr.Markdown(value="""
        <h1 style='text-align: center;'> Serial Experiments Lain </h1>
        <h2 style='text-align: center;'> So-VITS-SVC 4.1-Stable WebUI 推理&训练 v2023.8.10 </h2>
        """)
    # todo
    gr.Markdown(value="""
        ## 第一次使用前先点击“加载模型”
    
        ### 本页面仓库：[SuCicada/so-vits-svc](https://github.com/SuCicada/so-vits-svc/tree/4.1-Stable)
        
        ### 模型地址：[SuCicada/Lain-so-vits-svc-4.1](https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1)
        
        ### 我们的 Lain 中文交流频道：https://discord.gg/AS43RhmE
        <br>
        注意：本页面是为了推理Lain而准备的。其他完整功能请使用整合包，
        
        代码参考：bilibili@麦哲云
        
        [整合包，使用文档和常见报错解答](https://www.yuque.com/umoubuton/ueupp5)
        """)
    with gr.Tabs():
        with gr.TabItem("load model") as inference_tab:
            mode_caption = gr.Markdown(value=f"""
                {current_mode}，
            """)
            with gr.Row():
                loadckpt = gr.Button("加载模型", variant="primary")
                unload = gr.Button("卸载模型", variant="secondary")
            with gr.Row():
                model_message = gr.Textbox(label="Output Message")
                sid = gr.Dropdown(label="So-VITS说话人", value="lain")

            with gr.Row():
                enhance = gr.Checkbox(
                    label="是否使用NSF_HIFIGAN增强，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭",
                    value=False)
                only_diffusion = gr.Checkbox(
                    label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，不建议使用",
                    value=False)
            with gr.Row():
                diffusion_method = gr.Dropdown(label="扩散模型采样器",
                                               choices=["dpm-solver++", "dpm-solver", "pndm", "ddim", "unipc"],
                                               value="dpm-solver++")
                diffusion_speedup = gr.Number(label="扩散加速倍数，默认为10倍", value=10)

            gr.Markdown(value="""**下面参数已经默认设置好了**""")

            with gr.Row():
                choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value=default_model_path)
                model_branch = gr.Textbox(label="模型编码器", placeholder="请先选择模型", interactive=False)
            with gr.Row():
                config_choice = gr.Dropdown(label="配置文件", choices=config_list, value=default_config_path)
                config_info = gr.Textbox(label="配置文件编码器", placeholder="请选择配置文件")
            gr.Markdown(value="""**请检查模型和配置文件的编码器是否匹配**""")
            with gr.Row():
                diff_choice = gr.Dropdown(label="（可选）选择扩散模型", choices=diff_list, value=default_diff_model_path,
                                          interactive=True)
                diff_config_choice = gr.Dropdown(label="扩散模型配置文件", choices=diff_config_list,
                                                 value=default_diff_config_path, interactive=True)
            cluster_choice = gr.Dropdown(label="（可选）选择聚类模型/特征检索模型", choices=cluster_list,
                                         value=default_cluster_model_path)
            refresh = gr.Button("刷新选项")
            using_device = gr.Dropdown(label="推理设备，默认为自动选择", choices=["Auto", *cuda.keys(), "cpu"],
                                       value="Auto")

            inference_tab.select(refresh_options, [],
                                 [choice_ckpt, config_choice, cluster_choice, diff_choice, diff_config_choice])
            choice_ckpt.change(auto_load, [choice_ckpt], [model_branch, config_choice, config_info])
            config_choice.change(load_json_encoder, [config_choice, choice_ckpt], [config_info])
            diff_choice.change(auto_load_diff, [diff_choice], [diff_config_choice])
            refresh.click(refresh_options, [],
                          [choice_ckpt, config_choice, cluster_choice, diff_choice, diff_config_choice, mode_caption])

            gr.Markdown(value="""
                请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                """)

        with gr.TabItem("推理") as inference_tab:
            with gr.Tabs():
                with gr.TabItem("文字转语音"):
                    gr.Markdown("""
                        文字转语音（TTS）说明：使用edge_tts服务生成音频，并转换为So-VITS模型音色。
                    """)
                    text_input = gr.Textbox(label="在此输入需要转译的文字（建议打开自动f0预测）", lines=5,
                                            value="人はみな、繋がっている。")
                    with gr.Row():
                        # tts_gender = gr.Radio(label="说话人性别", choices=["男", "女"], value="男")
                        # tts_lang = gr.Dropdown(label="选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES,
                        #                        value="Auto")
                        tts_engine = gr.components.Dropdown(["edge-tts", "gtts", ], value="edge-tts",
                                                            label="tts engine")
                        language = gr.components.Dropdown(["ja", "en", "zh", ], value="ja",
                                                          label="language")
                    with gr.Row():
                        tts_rate = gr.Slider(label="TTS语音变速（倍速相对值）", minimum=0, maximum=5, value=1, step=0.1)
                        # tts_volume = gr.Slider(label="TTS语音音量（相对值）", minimum=-1, maximum=1.5, value=0, step=0.1)

                    vc_tts_submit = gr.Button("文本转语音", variant="primary")
                    with gr.Column():
                        origin_output = gr.Audio(label="origin voice")
                    tts_target_output: gr.Audio = gr.Audio(label="lain voice")

                with gr.TabItem("单个音频上传"):
                    vc_input3 = gr.Audio(label="单个音频上传", type="filepath", source="upload")
                    with gr.Row():
                        with gr.Column():
                            use_microphone = gr.Checkbox(label="使用麦克风输入")
                            vc_submit = gr.Button("音频转换", variant="primary")
                        vc_output1 = gr.Textbox(label="Output Message")
                    vc_output2 = gr.Audio(label="Output Audio")

                with gr.TabItem("批量音频上传"):
                    vc_batch_files = gr.Files(label="批量音频上传", file_types=["audio"], file_count="multiple")
                    vc_batch_submit = gr.Button("批量转换", variant="primary")

            with gr.Row():
                auto_f0 = gr.Checkbox(
                    label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会跑调）",
                    value=True)
                f0_predictor = gr.Radio(label="f0预测器选择（如遇哑音可以更换f0预测器解决，crepe为原F0使用均值滤波器）",
                                        choices=f0_options, value="rmvpe")
                cr_threshold = gr.Number(
                    label="F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音",
                    value=0.05)
            with gr.Row():
                vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                cluster_ratio = gr.Number(
                    label="聚类模型/特征检索混合比例，0-1之间，默认为0不启用聚类或特征检索，能提升音色相似度，但会导致咬字下降",
                    value=0.8)
                k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100,
                                   minimum=1, maximum=1000)
            with gr.Row():
                output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value="wav")
                enhancer_adaptive_key = gr.Number(label="使NSF-HIFIGAN增强器适应更高的音域(单位为半音数)|默认为0",
                                                  value=0)
                slice_db = gr.Number(label="切片阈值", value=-50)
                cl_num = gr.Number(label="音频自动切片，0为按默认方式切片，单位为秒/s，爆显存可以设置此处强制切片",
                                   value=0)

            with gr.Accordion("高级设置（一般不需要动）", open=False):
                noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                pad_seconds = gr.Number(
                    label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现",
                    value=0.01)
                lg_num = gr.Number(
                    label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
                    value=1)
                lgr_num = gr.Number(
                    label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭",
                    value=0.75)
                second_encoding = gr.Checkbox(
                    label="二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False)
                loudness_envelope_adjustment = gr.Number(
                    label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value=0)
                use_spk_mix = gr.Checkbox(label="动态声线融合，需要手动编辑角色混合轨道，没做完暂时不要开启", value=False,
                                          interactive=False)
            # with gr.Row():
            # interrupt_button = gr.Button("中止转换", variant="danger")

        loadckpt.click(load_model_func,
                       [choice_ckpt, cluster_choice, config_choice, enhance, diff_choice, diff_config_choice,
                        only_diffusion, use_spk_mix, using_device, diffusion_method, diffusion_speedup, cl_num],
                       [model_message, sid, cl_num, k_step,
                        model_branch, config_info])
        unload.click(model_empty_cache, [], [sid, model_message])
        use_microphone.change(source_change, [use_microphone], [vc_input3])
        vc_tts_submit.click(tts_fn,
                            [text_input, tts_engine, language, tts_rate, output_format, sid, vc_transform,
                             auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num,
                             f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding,
                             loudness_envelope_adjustment], [origin_output, tts_target_output])
        vc_submit.click(vc_fn,
                        [output_format, sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
                         pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold,
                         k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_batch_submit.click(vc_batch_fn,
                              [output_format, sid, vc_batch_files, vc_transform, auto_f0, cluster_ratio, slice_db,
                               noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key,
                               cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment],
                              [vc_output1])
        # interrupt_button.click(fn=None, inputs=None, outputs=None, cancels=[vc_event])

        with gr.TabItem("小工具/实验室特性"):
            gr.Markdown(value="""
                        ### So-vits-svc 4.1 小工具/实验室特性
                        提供了一些有趣或实用的小工具，可以自行探索
                        """)
            with gr.Tabs():
                with gr.TabItem("静态声线融合"):
                    gr.Markdown(value="""
                        <font size=2> 介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线 
                                          注意：
                                          1.该功能仅支持单说话人的模型
                                          2.如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
                                          3.保证所有待混合模型的config.json中的model字段是相同的
                                          4.输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
                                          5.批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
                                          6.混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
                                          7.混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
                                          8.凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会
                        </font>
                        """)
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple")
                    mix_model_output1 = gr.Textbox(
                        label="混合比例调整，单位/%",
                        interactive=True
                    )
                    mix_mode = gr.Radio(choices=["凸组合", "线性组合"], label="融合模式", value="凸组合",
                                        interactive=True)
                    mix_submit = gr.Button("声线融合启动", variant="primary")
                    mix_model_output2 = gr.Textbox(
                        label="Output Message"
                    )
                with gr.TabItem("onnx转换"):
                    gr.Markdown(value="""
                        提供了将.pth模型（批量）转换为.onnx模型的功能
                        源项目本身自带转换的功能，但不支持批量，操作也不够简单，这个工具可以支持在WebUI中以可视化的操作方式批量转换.onnx模型
                        有人可能会问，转.onnx模型有什么作用呢？相信我，如果你问出了这个问题，说明这个工具你应该用不上

                        ### Step 1: 
                        在整合包根目录下新建一个"checkpoints"文件夹，将pth模型和对应的json配置文件按目录分别放置到checkpoints文件夹下
                        看起来应该像这样：
                        checkpoints
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        ├───xxxx
                        │   ├───xxxx.pth
                        │   └───xxxx.json
                        └───……
                        """)
                    pth_dir_msg = gr.Textbox(label="识别待转换模型",
                                             placeholder="请将模型和配置文件按上述说明放置在正确位置")
                    pth_dir_identify_btn = gr.Button("识别", variant="primary")
                    gr.Markdown(value="""
                        ### Step 2:
                        识别正确后点击下方开始转换，转换一个模型可能需要一分钟甚至更久
                        """)
                    pth2onnx_btn = gr.Button("开始转换", variant="primary")
                    pth2onnx_msg = gr.Textbox(label="输出信息")

                with gr.TabItem("智能音频切片"):
                    gr.Markdown(value="""
                        该工具可以实现对音频的切片，无需调整参数即可完成符合要求的数据集制作。
                        数据集要求的音频切片约在2-15秒内，用传统的Slicer-GUI切片工具需要精准调参和二次切片才能符合要求，该工具省去了上述繁琐的操作，只要上传原始音频即可一键制作数据集。
                    """)
                    with gr.Row():
                        raw_audio_path = gr.Textbox(label="原始音频文件夹",
                                                    placeholder="包含所有待切片音频的文件夹，示例: D:\干声\speakers")
                        load_raw_audio_btn = gr.Button("加载原始音频", variant="primary")
                    load_raw_audio_output = gr.Textbox(label="输出信息")
                    raw_audio_dataset = gr.Textbox(label="音频列表", value="")
                    slicer_output_dir = gr.Textbox(label="输出目录",
                                                   placeholder="选择输出目录（不要和输入音频是同一个文件夹）")
                    with gr.Row():
                        process_method = gr.Radio(label="对过短音频的处理方式",
                                                  choices=["丢弃", "将过短音频整合为长音频"], value="丢弃")
                        max_sec = gr.Number(label="切片的最长秒数", value=15)
                        min_sec = gr.Number(label="切片的最短秒数", value=2)
                    slicer_btn = gr.Button("开始切片", variant="primary")
                    slicer_output_msg = gr.Textbox(label="输出信息")

                    mix_model_path.change(updata_mix_info, [mix_model_path], [mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button, mix_model_path],
                                                   [mix_model_path, mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1, mix_mode], [mix_model_output2])
                    pth_dir_identify_btn.click(pth_identify, [], [pth_dir_msg])
                    pth2onnx_btn.click(onnx_export, [], [pth2onnx_msg])
                    load_raw_audio_btn.click(load_raw_audio, [raw_audio_path],
                                             [load_raw_audio_output, raw_audio_dataset])
                    slicer_btn.click(slicer_fn, [raw_audio_path, slicer_output_dir, process_method, max_sec, min_sec],
                                     [slicer_output_msg])

                with gr.TabItem("模型压缩工具"):
                    gr.Markdown(value="""
                        该工具可以实现对模型的体积压缩，在**不影响模型推理功能**的情况下，将原本约600M的So-VITS模型压缩至约200M, 大大减少了硬盘的压力。
                        **注意：压缩后的模型将无法继续训练，请在确认封炉后再压缩。**
                        将模型文件放置在logs/44k下，然后选择需要压缩的模型
                    """)
                    model_to_compress = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
                    fp16_compress = gr.Checkbox(label="使用 fp16 压缩", value=False)
                    compress_model_btn = gr.Button("压缩模型", variant="primary")
                    compress_model_output = gr.Textbox(label="输出信息", value="")

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])

                with gr.TabItem("模型发布打包/安装"):
                    gr.Markdown(value="""
                        如果你想将你的模型分享给他人，请使用该工具对模型进行打包。
                        该工具可以自动生成正确的配置文件，确保你在打包过程中不出现任何遗漏和错误，接收到使用该工具打包的模型后，也可以用该工具进行自动安装。
                    """)
                    with gr.Tabs():
                        with gr.TabItem("安装"):
                            with gr.Row():
                                model_to_install = gr.Textbox(label="模型压缩包路径",
                                                              placeholder="示例：D:\Downloads\model_packing.zip")
                                install_model_btn = gr.Button("安装", variant="primary")
                            install_output = gr.Textbox(label="输出信息", value="")
                        with gr.TabItem("打包"):
                            with gr.Row():
                                model_to_pack = gr.Dropdown(label="选择要打包的模型", choices=ckpt_list, value="")
                                model_config = gr.Dropdown(label="选择要打包的模型配置文件", choices=config_list,
                                                           value="", interactive=True)
                                speaker_name = gr.Textbox(label="模型说话人名称",
                                                          placeholder="该模型的说话人名称，仅限数字字母下划线，如模型中有多说话人，请用逗号分割，例如：spk1,spk2,spk3",
                                                          value="")
                            with gr.Row():
                                diff_to_pack = gr.Dropdown(label="（可选）选择要打包的扩散模型", choices=diff_list,
                                                           value="no_diff")
                                cluster_to_pack = gr.Dropdown(label="（可选）选择要打包的聚类或特征检索模型",
                                                              choices=cluster_list, value="no_cluster")
                            packing_btn = gr.Button("开始打包", variant="primary")
                            packing_output_msg = gr.Textbox(label="输出信息")

                    model_to_pack.change(pack_autoload, [model_to_pack], [model_config, speaker_name])
                    packing_btn.click(release_packing,
                                      [model_to_pack, model_config, speaker_name, diff_to_pack, cluster_to_pack],
                                      [packing_output_msg])
                    install_model_btn.click(release_install, [model_to_install], [install_output])


def main():
    if args.debug:
        port = networking.get_first_available_port(
            networking.INITIAL_PORT_VALUE,
            networking.INITIAL_PORT_VALUE + networking.TRY_NUM_PORTS,
        )
        original_path = sys.argv[0]
        from gradio import utils as gradio_utils
        abs_original_path = gradio_utils.abspath(original_path)
        path = os.path.normpath(original_path)
        path = path.replace("/", ".")
        path = path.replace("\\", ".")
        filename = os.path.splitext(path)[0]

        # gradio_folder = Path(inspect.getfile(gradio)).parent
        abs_parent: str = str(abs_original_path.parent)

        print("filename", filename, abs_parent)
        print(f"http://127.0.0.1:{port}")
        uvicorn.run(f"{filename}:app.app", reload=True, reload_dirs=[abs_parent], port=port, log_level="warning")
    else:
        app.queue(concurrency_count=1022, max_size=2044) \
            .launch(root_path=args.root_path,
                    server_name="0.0.0.0",
                    server_port=args.port,
                    share=args.share)


if __name__ == '__main__':
    # app()
    main()
