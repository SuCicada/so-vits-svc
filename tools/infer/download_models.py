import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import dotenv
import requests
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import models_info
from tools.file_util import get_root_project

dotenv.load_dotenv()


def wget(url, headers, filename):
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()  # 确保请求成功
    total_size = int(response.headers.get('content-length', 0))  # 获取文件总大小
    # 使用tqdm显示进度条
    with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def wget_if_not_exist(url: str, destination: str):
    # 检查目标文件是否存在
    # print(url, destination)
    if not os.path.exists(destination):
        try:
            headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
            # request = urllib.request.Request(url, headers=headers)
            # urllib.request.urlretrieve(request, destination)
            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination), exist_ok=True)
            wget(url, headers, destination)
            # urllib.request.urlretrieve(url, destination)
            print(f"Downloaded {url} => {destination}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"File {destination} already exists. Skipping download.")


def download_models(model_info: models_info.ModelsInfo):
    for model in model_info.get_models_list():
        remote_model_path = os.path.join(model_info.remote_dir, model)
        url = f"https://huggingface.co/{model_info.hf_repo}/resolve/main/{remote_model_path}"
        r = get_root_project()

        destination = r / "models" / model
        # os.makedirs(r / "tmp", exist_ok=True)
        # destination = r / "tmp" / model
        # destination = Path(tmpdir) / model
        wget_if_not_exist(url, str(destination))


model_info = models_info.LainV2
# with tempfile.TemporaryDirectory() as tmpdir:
download_models(model_info)
