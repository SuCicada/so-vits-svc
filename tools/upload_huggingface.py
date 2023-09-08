import os
from huggingface_hub import login
from huggingface_hub import HfApi
from dotenv import load_dotenv
from file_util import get_root_project

root_project = get_root_project()
directory_path = root_project / "models_archive/4.1/202309"
remote_dir = "202309/models"

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

update_list = [
    (root_project / "release_packs/lain_release.zip", f"{remote_dir}/release_packs/lain_release.zip"),
]


def sacn_upload_list(_dir):
    with os.scandir(_dir) as entries:
        for entry in entries:
            if entry.is_dir():  # and entry.name == "diffusion":
                sacn_upload_list(entry.path)
            elif entry.is_file():
                remote_path = entry.path.replace(str(directory_path), remote_dir)
                update_list.append((entry.path, remote_path))


def upload_from_list():
    for local, remote in update_list:
        print(f"uploading {local} => {remote}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=os.getenv("HF_REPO_ID"),
            repo_type="model",
)

sacn_upload_list(directory_path)
upload_from_list()
