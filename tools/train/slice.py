# ==================================================================================
# reference from
# https://www.yuque.com/umoubuton/ueupp5/sdahi7m5m6r0ur1r#zMbpu
# ==================================================================================

import sys
from pathlib import Path
import librosa
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
print(sys.path)

from auto_slicer import AutoSlicer

def slicer_fn(input_dir, output_dir, process_method, max_sec, min_sec):
    if output_dir == "":
        return "请先选择输出的文件夹"
    if output_dir == input_dir:
        return "输出目录不能和输入目录相同"
    slicer = AutoSlicer()
    if os.path.exists(output_dir) is not True:
        os.makedirs(output_dir)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(".wav"):
            slicer.auto_slice(filename, input_dir, output_dir, max_sec)
    if process_method == "丢弃":
        for filename in tqdm(os.listdir(output_dir)):
            if filename.endswith(".wav"):
                filepath = os.path.join(output_dir, filename)
                audio, sr = librosa.load(filepath, sr=None, mono=False)
                if librosa.get_duration(y=audio, sr=sr) < min_sec:
                    os.remove(filepath)
                    print(f"too short, remove {filename}")
    elif process_method == "将过短音频整合为长音频":
        slicer.merge_short(output_dir, max_sec, min_sec)
    file_count, max_duration, min_duration, orig_duration, final_duration = slicer.slice_count(input_dir, output_dir)
    hrs = int(final_duration / 3600)
    mins = int((final_duration % 3600) / 60)
    sec = format(float(final_duration % 60), '.2f')
    rate = format(100 * (final_duration / orig_duration), '.2f') if orig_duration != 0 else 0
    rate_msg = f"为原始音频时长的{rate}%" if rate != 0 else "因未知问题，无法计算切片时长的占比"
    return f"成功将音频切分为{file_count}条片段，其中最长{max_duration}秒，最短{min_duration}秒，切片后的音频总时长{hrs:02d}小时{mins:02d}分{sec}秒，{rate_msg}"


import os

slicer_fn("/Users/peng/PROGRAM/GitHub/lain_data/output/good",
          "/Users/peng/PROGRAM/GitHub/so-vits-svc/dataset_raw/lain",
          "丢弃", 15, 2)
