import asyncio
import base64
import io
import os
from io import BytesIO
from typing_extensions import deprecated

from pydub import AudioSegment
import numpy
import requests
import json
from tools import audio_utils


def text_to_audio(text, tts_engine="gtts", language="ja", speed=1) -> (int, numpy.ndarray):
    url = os.getenv("TTSHUB_API_URL")
    payload = json.dumps({
        "tts_engine": tts_engine,
        "text": text,
        "language": language,
        "speed": speed
    })
    headers = {
        'Content-Type': 'application/json'
    }
    print("url:", url)
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"TTSHub API Error: {response.status_code}")
    result = response.json()
    sampling_rate,wav_bytes_base64 = result["sampling_rate"], result["audio"]
    wav_bytes = base64.b64decode(wav_bytes_base64)
    numarr: numpy.ndarray = numpy.frombuffer(wav_bytes, dtype=numpy.int16)
    return sampling_rate, numarr

@deprecated("")
def tts_with_edge_tts(text, language, _rate) -> BytesIO:
    import edge_tts
    voice = {
        "ja": "ja-JP-NanamiNeural",
        "en": "en-US-AriaNeural",
        "zh": "zh-CN-XiaoxiaoNeural"
    }[language]

    _rate = f"+{int((_rate - 1) * 100)}%" if _rate >= 1 else f"{int((1 - _rate) * 100)}%"

    async def _write():
        file_in_memory = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice, rate=_rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file_in_memory.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(f"WordBoundary: {chunk}")
        return file_in_memory

    return asyncio.run(_write())


@deprecated("")
def tts_with_gtts(text, language) -> BytesIO:
    lang = {
        "ja": "ja",
        "en": "en",
        "zh": "zh-CN"
    }[language]

    from gtts import gTTS
    stream = BytesIO()
    tts = gTTS(text, lang=lang, slow=False)
    tts.write_to_fp(stream)
    return stream


@deprecated("")
def mp3_to_wav(mp3_bytes) -> (int, bytes):
    # mp3 to wav
    mp3 = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    sampling_rate = mp3.frame_rate
    wav = io.BytesIO()
    mp3.export(wav, format="wav")
    wav_bytes = wav.getvalue()
    return sampling_rate, wav_bytes


@deprecated("")
def text_to_wav(text, tts_engine="gtts", language="ja", rate=1) -> (int, bytes):
    text = text.strip()
    stream: BytesIO
    print("use tts_engine", tts_engine)
    print("text:", text)

    if tts_engine == "gtts":
        stream = tts_with_gtts(text, language)
    elif tts_engine == "edge-tts":
        stream = tts_with_edge_tts(text, language, rate)
    else:
        raise Exception("tts_engine not support")

    mp3_bytes = stream.getvalue()
    sampling_rate, wav_bytes = mp3_to_wav(mp3_bytes)
    return sampling_rate, wav_bytes

@deprecated("")
def text_to_audio1(text, tts_engine="gtts", language="ja", rate=1) -> (int, numpy.ndarray):
    sampling_rate, wav_bytes = text_to_wav(text, tts_engine, language, rate)
    wav: numpy.ndarray = numpy.frombuffer(wav_bytes, dtype=numpy.int16)
    if tts_engine == "gtts":
        if rate != 1:
            wav = audio_utils.modify_speed(sampling_rate, wav, rate)
    return sampling_rate, wav
