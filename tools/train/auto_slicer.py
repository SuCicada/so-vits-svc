# ==================================================================================
# reference from
# https://www.yuque.com/umoubuton/ueupp5/sdahi7m5m6r0ur1r#zMbpu
# ==================================================================================

import os
from typing import final
import numpy as np
import librosa
import soundfile as sf
from slicer2 import Slicer

class AutoSlicer:
    def __init__(self):
        self.slicer_params = {
            "threshold": -40,
            "min_length": 5000,
            "min_interval": 300,
            "hop_size": 10,
            "max_sil_kept": 500,
        }
        self.original_min_interval = self.slicer_params["min_interval"]

    def auto_slice(self, filename, input_dir, output_dir, max_sec):
        audio, sr = librosa.load(os.path.join(input_dir, filename), sr=None, mono=False)
        slicer = Slicer(sr=sr, **self.slicer_params)
        chunks = slicer.slice(audio)
        files_to_delete = []
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            output_filename = f"{os.path.splitext(filename)[0]}_{i}"
            output_filename = "".join(c for c in output_filename if c.isascii() or c == "_") + ".wav"
            output_filepath = os.path.join(output_dir, output_filename)
            sf.write(output_filepath, chunk, sr)
            #Check and re-slice audio that more than max_sec.
            while True:
                new_audio, sr = librosa.load(output_filepath, sr=None, mono=False)
                if librosa.get_duration(y=new_audio, sr=sr) <= max_sec:
                    break
                self.slicer_params["min_interval"] = self.slicer_params["min_interval"] // 2
                if self.slicer_params["min_interval"] >= self.slicer_params["hop_size"]:
                    new_chunks = Slicer(sr=sr, **self.slicer_params).slice(new_audio)
                    for j, new_chunk in enumerate(new_chunks):
                        if len(new_chunk.shape) > 1:
                            new_chunk = new_chunk.T
                        new_output_filename = f"{os.path.splitext(output_filename)[0]}_{j}.wav"
                        sf.write(os.path.join(output_dir, new_output_filename), new_chunk, sr)
                    files_to_delete.append(output_filepath)
                else:
                    break
            self.slicer_params["min_interval"] = self.original_min_interval
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)

    def merge_short(self, output_dir, max_sec, min_sec):
        short_files = []
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if filename.endswith(".wav"):
                audio, sr = librosa.load(filepath, sr=None, mono=False)
                duration = librosa.get_duration(y=audio, sr=sr)
                if duration < min_sec:
                    short_files.append((filepath, audio, duration))
        short_files.sort(key=lambda x: x[2], reverse=True)
        merged_audio = []
        current_duration = 0
        for filepath, audio, duration in short_files:
            if current_duration + duration <= max_sec:
                merged_audio.append(audio)
                current_duration += duration
                os.remove(filepath)
            else:
                if merged_audio:
                    output_audio = np.concatenate(merged_audio, axis=-1)
                    if len(output_audio.shape) > 1:
                        output_audio = output_audio.T
                    output_filename = f"merged_{len(os.listdir(output_dir))}.wav"
                    sf.write(os.path.join(output_dir, output_filename), output_audio, sr)
                    merged_audio = [audio]
                    current_duration = duration
                    os.remove(filepath)
        if merged_audio and current_duration >= min_sec:
            output_audio = np.concatenate(merged_audio, axis=-1)
            if len(output_audio.shape) > 1:
                output_audio = output_audio.T
            output_filename = f"merged_{len(os.listdir(output_dir))}.wav"
            sf.write(os.path.join(output_dir, output_filename), output_audio, sr)

    def slice_count(self, input_dir, output_dir):
        orig_duration = final_duration = 0
        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                _audio, _sr = librosa.load(os.path.join(input_dir, file), sr=None, mono=False)
                orig_duration += librosa.get_duration(y=_audio, sr=_sr)
        wav_files = [file for file in os.listdir(output_dir) if file.endswith(".wav")]
        num_files = len(wav_files)
        max_duration = -1
        min_duration = float("inf")
        for file in wav_files:
            file_path = os.path.join(output_dir, file)
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=audio, sr=sr)
            final_duration += float(duration)
            if duration > max_duration:
                max_duration = float(duration)
            if duration < min_duration:
                min_duration = float(duration)
        return num_files, max_duration, min_duration, orig_duration, final_duration


