import json
import requests

text = """
おはようございます！何かお力になれることがあれば、おっしゃってください。
"""
req_json = {
    'text': text,
    "speed":1.2,
}
response = requests.post('http://localhost:7100/audio/so_vits_svc', json=req_json)

# json_data = pickle.loads(response.content)
json_data = json.loads(response.headers["Response-Data"])
print(json_data)
# audio = json_data['audio']
audio = response.content
# audio = numpy.frombuffer(audio, dtype=numpy.int16)
sampling_rate = json_data['sampling_rate']
# print(audio, sampling_rate)
# sounddevice.play(audio, sampling_rate, blocking=True)
import pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, output=True)
stream.write(audio)
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
