import io
import tempfile
from typing import Union, Tuple

import numpy
import soundfile
from pydub import AudioSegment
from . import file_util


def numpy_to_tmp_file(sampling_rate, audio_data: numpy.ndarray) -> str:
    # with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
    _, temp_path = tempfile.mkstemp()
    soundfile.write(temp_path, audio_data, sampling_rate, format="wav")
    return temp_path


def numpy_to_mem_file(sampling_rate: int, audio_data: numpy.ndarray) -> io.BytesIO:
    # from scipy.io.wavfile import write
    wav_file = io.BytesIO()
    # write(wav_file, sampling_rate, audio_data)
    # return wav_file
    print("audio_data.dtype", audio_data.dtype)
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_data = (audio_data * 32767).astype(np.int16)
    soundfile.write(wav_file, audio_data, sampling_rate, format="wav")
    # write(wav_file, sampling_rate, audio_data)
    wav_file.seek(0)  # Important: reset the file pointer to the beginning of the file
    return wav_file


def modify_speed(sampling_rate, audio_data: numpy.ndarray, speed=1.25) -> numpy.ndarray:
    print("modify_speed", speed)
    # wav = numpy_to_mem_file(audio_data, sampling_rate, )
    # with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
    with file_util.MyNamedTemporaryFile() as temp_path:
        # temp_path = temp_file.name
        soundfile.write(temp_path, audio_data, sampling_rate, format="wav")
        audio = AudioSegment.from_file(temp_path)
        # audio = audio.set_channels(1).set_sample_width(2)  # Ensure the audio is mono and 16-bit
        audio = audio.speedup(playback_speed=speed)
        numpy_array = np.array(audio.get_array_of_samples())

        # sounddevice.play(numpy_array, sampling_rate, blocking=True)

        # return numpy_array
        # audio = speed_change(audio, speed)

        # import pyrubberband as pyrb
        # import soundfile as sf
        # y, sr = sf.read(temp_path)
        # # Play back at extra low speed
        # y_stretch = pyrb.time_stretch(y, sr, speed)
        # # Play back extra low tones
        # y_shift = pyrb.pitch_shift(y, sr, speed)
        # with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file2:
        with file_util.MyNamedTemporaryFile() as temp_path2:
            # sf.write(temp_file2.name, y_stretch, sr, format='wav')
            soundfile.write(temp_path2, numpy_array, sampling_rate, format="wav")
            _, audio = wav_to_numpy(temp_path2)
            return audio
            # numpy_array = np.array(audio.get_array_of_samples())
            # return numpy_array


def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    res = sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
    return res


import wave
import numpy as np


def wav_to_numpy(file_path):
    with wave.open(file_path, "rb") as wav_file:
        # Get the audio file parameters
        channels = wav_file.getnchannels()
        # sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        # Read the audio data from the WAV file
        audio_data = wav_file.readframes(num_frames)
        # Convert the audio data to a NumPy array
        audio_numpy = np.frombuffer(audio_data, dtype=np.int16)
        # Reshape the NumPy array based on the number of channels
        audio_numpy = np.reshape(audio_numpy, (num_frames, channels))
        return sample_rate, audio_numpy

    # if __name__ == '__main__':


# audio_numpy, sample_rate = wav_to_numpy("tmpmom974j6.wav")
# audio_numpy = modify_speed(audio_numpy, sample_rate, 1.1)
#
# # fast_sound = speed_change(sound, 1.1)
# soundfile.write("out.wav", audio_numpy, sample_rate, format="wav")


def merge_audio(audio_file1: Union[str, Tuple[int, np.ndarray]],
                audio_file2: Union[str, Tuple[int, np.ndarray]]) -> (int, numpy.ndarray):
    if isinstance(audio_file1, tuple):
        audio_file1 = numpy_to_mem_file(audio_file1[0], audio_file1[1])
    if isinstance(audio_file2, tuple):
        audio_file2 = numpy_to_mem_file(audio_file2[0], audio_file2[1])

    audio1 = AudioSegment.from_file(audio_file1)
    audio2 = AudioSegment.from_file(audio_file2)
    # 检查两个音频文件的长度，选择最长的音频作为基准
    max_length = max(len(audio1), len(audio2))
    # 将两个音频文件进行同步
    audio1 = audio1[:max_length]
    audio2 = audio2[:max_length]
    # 将两个音频文件进行合并
    combined = audio1.overlay(audio2)
    # 保存合并后的音频文件
    # with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file2:
    with file_util.MyNamedTemporaryFile() as temp_path:
        combined.export(temp_path, format="wav")
        combined_sampling_rate, combined_audio = wav_to_numpy(temp_path)
        return combined_sampling_rate, combined_audio
