import json
import json
import os
import sys

from flask import make_response, Flask, request
from flask_cors import CORS
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools.audio_utils import numpy_to_mem_file
from tools.infer_base import SvcInfer

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from typing import Dict, Any
#
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app = Flask(__name__)
CORS(app)

# so_vits_svc = SoVitsSvcTTS()
# TTS_Models = {}
#
# for name, model in character_model_map.items():
#     TTS_Models[name] = SoVitsSvcTTS(model)
svc_infer: SvcInfer = None


# class AudioParams(BaseModel):
#     text: str
    # speaker: int


# @app.post("/audio/so_vits_svc/{id}")
# def audio_data(id):
def new_svc_infer():
    return SvcInfer(
        model_path="models/G_256800_infer.pth",
        config_path="models/config.json",
        cluster_model_path="models/kmeans_10000.pt",
        hubert_model_path="models/checkpoint_best_legacy_500.pt"
    )


@app.post("/audio/so_vits_svc")
def audio():
    global svc_infer
    if svc_infer is None:
        svc_infer = new_svc_infer()
    params: dict = request.json
    text = params['text'].strip()
    # speaker = params.get('speaker',"mikisayaka").strip()
    speed = params.get('speed')
    tts_engine = params.get('tts_engine')
    # params['speaker']
    # print(speaker, text)
    # tts = TTS_Models[speaker]
    # sampling_rate, audio_data = tts.get_audio(text)
    # audio_data: numpy.ndarray = audio_data
    # byte_data = pickle.dumps({
    #     "status": "ok",
    #     "sampling_rate": sampling_rate
    # })
    # "audio": audio_data.tobytes(),
    if not speed:
        speed = 1.1
    if not tts_engine:
        tts_engine = "gtts"
    (origin_sampling_rate, origin_audio), (target_sampling_rate, target_audio) = svc_infer.get_audio(
        tts_engine=tts_engine,
        text=text,
        speed=speed,
        language="ja",
    )

    response_data = {
        "status": "ok",
        "sampling_rate": target_sampling_rate
    }
    # byte_data = audio_data.tobytes()
    # from scipy.io.wavfile import write
    # with io.BytesIO() as wav_file:
    #     write(wav_file, sampling_rate, audio_data)

    # if speed:
    #     audio_data = modify_speed(audio_data, sampling_rate, float(speed))
    #
    wav_file = numpy_to_mem_file(target_sampling_rate, target_audio)
    byte_data = wav_file.getvalue()

    response = make_response(byte_data)
    # open("hello.wav", "wb").write(audio_data)
    # from scipy.io.wavfile import write
    # write("hello.wav", sampling_rate, audio_data)
    # sounddevice.play(audio, sampling_rate, blocking=True)
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers['Response-Data'] = json.dumps(response_data)
    return response


def main():
    port = os.environ.get('PORT', 7100)
    app.run(host="0.0.0.0", port=port, debug=True, threaded=False)
main()
# def main():
#     uvicorn.run(app, host="0.0.0.0",port=7100, reload=True)
# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0",port=7100, reload=True)
# if __name__ == '__main__':
#     main()
