import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Callable

import uvicorn
import gradio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tools import file_util
from tools.infer_base import SvcInfer

# from tools.lain_gradio import get_svc_infer

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from tools.file_util import get_root_project
from tools.infer import models_info

from tools.audio_utils import numpy_to_mem_file

model_info = models_info.LainV2


def get_model_config():
    return model_info.to_svc_config(f"{get_root_project()}/models")


app = FastAPI()


def main():
    # svc_infer = SvcInfer(get_model_config())
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help='port')
    args = parser.parse_args()

    # class AudioParams(BaseModel):
    #     text: str
    #     speaker: str = "mikisayaka"

    from lain_gradio import app as gradio_app

    gradio_fastapi_app = gradio.routes.App.create_app(gradio_app)
    app.mount("/", gradio_fastapi_app)

    # @app.post("/audio/so_vits_svc")
    # async def _generate_audio(params: Dict[str, Any]):
    #     return generate_audio(params, svc_infer)
    add_server_api(app)
    port = args.port
    if not port:
        port = os.environ.get('PORT', 17861)
    uvicorn.run("lain_server:app", host="0.0.0.0", port=port, reload=True)


def add_server_api(app: FastAPI, get_svc_infer: Callable[..., SvcInfer]):
    print("add_server_api")

    @app.post("/svcapi/get_audio")
    async def generate_audio(request: Request):
        params = await request.json()
        print("generate_audio", params)
        loop = asyncio.get_event_loop()

        _, (sampling_rate, audio_data) = await loop.run_in_executor(None,
                                                                    lambda: get_svc_infer().get_audio(**params))
        response_data = {
            "status": "ok",
            "sampling_rate": sampling_rate
        }
        headers = {"Response-Data": json.dumps(response_data)}

        data = numpy_to_mem_file(sampling_rate, audio_data)
        return StreamingResponse(data, media_type="audio/wav", headers=headers)

    @app.post("/svcapi/audio_to_audio")
    async def audio_to_audio(request: Request):
        params = await request.json()
        audio_base64: str = params.pop("audio")
        audio_bytes = base64.b64decode(audio_base64)
        with file_util.MyNamedTemporaryFile() as temp_path:
            Path(temp_path).write_bytes(audio_bytes)
            sampling_rate, audio_data = get_svc_infer().transform_audio_file(temp_path, **params)
        # print("generate_audio", params)
        # loop = asyncio.get_event_loop()
        # _, (sampling_rate, audio_data) = await loop.run_in_executor(None,
        # #                                                             lambda: get_svc_infer().get_audio(**params))
        # response_data = {
        #     "status": "ok",
        #     "sampling_rate": sampling_rate
        # }
        # headers = {"Response-Data": json.dumps(response_data)}
        data = numpy_to_mem_file(sampling_rate, audio_data)
        res_audio_base64 = base64.b64encode(data.getvalue()).decode()
        # return StreamingResponse(data, media_type="audio/wav", headers=headers)
        return {
            "status": "ok",
            "sampling_rate": sampling_rate,
            "audio": res_audio_base64
        }

if __name__ == "__main__":
    main()
