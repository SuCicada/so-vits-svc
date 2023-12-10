import base64
import json
from pathlib import Path

import requests
import sounddevice
import soundfile

text = """
シミュレーション, パーソナルコンピュータ
"""
bytes = Path("/Users/peng/PROGRAM/GitHub/so-vits-svc/tools/origin.wav").read_bytes()
data_base64 = base64.b64encode(bytes).decode()
req_json = {
    # "data":[{
    # "text": text,
    # "tts_engine": "edge-tts",
    # "speed": 1,
    "audio": data_base64,
    # }]
}
url = 'http://localhost:17861/svcapi/audio_to_audio'
# url="https://lain.sucicada.top/so-vits-svc/svcapi/get_audio"
response = requests.post(url, json=req_json)
# json_data = pickle.loads(response.content)
# json_data = json.loads(response.headers["Response-Data"])
# print(json_data)
# audio = json_data['audio']
res = response.json()
audio = base64.b64decode(res['audio'])
sampling_rate = res['sampling_rate']
# audio = numpy.frombuffer(audio, dtype=numpy.int16)
# sampling_rate = json_data['sampling_rate']
# print(audio, sampling_rate)
with open("out.wav", "wb") as f:
    f.write(audio)
# sounddevice.play(audio, sampling_rate, blocking=True)
# soundfile.write("out.wav", sampling_rate, audio, format="wav")
# sounddevice.play(audio, sampling_rate, blocking=True)
from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file('out.wav')
play(sound)

import pydub
# import pyaudio
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, output=True)
# stream.write(audio)
# sounddevice.play(audio, sampling_rate, blocking=True)

# # Close stream and terminate PyAudio
# stream.stop_stream()
# stream.close()
# p.terminate()

# buffer = io.BytesIO()
# write("hello.wav", sampling_rate, audio)
# from scipy.io.wavfile import write
# with io.BytesIO() as wav_file:
#     write(wav_file, sampling_rate, audio)
#     wav_binary = wav_file.getvalue()

# open("hello.wav", "wb").write(wav_binary)
