import io

import numpy
import numpy as np
import soundfile
from pydub import AudioSegment


def numpy_to_mem_file(audio_data: numpy.ndarray, sampling_rate) -> io.BytesIO:
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


def modify_speed(audio_data: numpy.ndarray, sampling_rate, speed=1.25) -> numpy.ndarray:
    print("modify_speed", speed)
    wav = numpy_to_mem_file(audio_data, sampling_rate, )
    audio = AudioSegment.from_file(wav)
    # audio = audio.set_channels(1).set_sample_width(2)  # Ensure the audio is mono and 16-bit
    audio = audio.speedup(playback_speed=speed)
    numpy_array = np.array(audio.get_array_of_samples())
    return numpy_array
