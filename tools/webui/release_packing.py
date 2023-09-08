import ast
import os
import shutil
import sys
import zipfile
from pathlib import Path

import torch

root_project = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(root_project))

from compress_model import removeOptimizer
from tools.webui.webui_utils import Config, MODEL_TYPE

second_dir = root_project / "models"
diff_second_dir = second_dir / "diffusion"
ckpt_read_dir = second_dir
config_read_dir = second_dir
diff_read_dir = diff_second_dir


class ReleasePacker:
    def __init__(self, speaker, model):
        self.speaker = speaker
        self.model = model
        self.output_path = os.path.join("release_packs", f"{speaker}_release.zip")
        self.file_list = []

    def remove_temp(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and not filename.endswith(".zip"):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)

    def add_file(self, file_paths):
        self.file_list.extend(file_paths)

    def spk_to_dict(self):
        spk_string = self.speaker.replace('，', ',')
        spk_string = spk_string.replace(' ', '')
        _spk = spk_string.split(',')
        return {_spk: index for index, _spk in enumerate(_spk)}

    def generate_config(self, diff_model, config_origin):
        if not os.path.exists(config_origin):
            config_origin = os.path.join(config_read_dir, config_origin)
        if not os.path.exists(self.model):
            self.model = os.path.join(ckpt_read_dir, self.model)

        _config_origin = Config(config_origin, "json")
        _template = Config("release_packs/config_template.json", "json")
        _d_template = Config("release_packs/diffusion_template.yaml", "yaml")
        orig_config = _config_origin.read()
        config_template = _template.read()
        diff_config_template = _d_template.read()
        spk_dict = self.spk_to_dict()
        _net = torch.load(self.model, map_location='cpu')
        emb_dim, model_dim = _net['model'].get('emb_g.weight', torch.empty(0, 0)).size()
        vol_emb = _net['model'].get('emb_vol.weight')
        if vol_emb is not None:
            config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True
        # Keep the spk_dict length same as emb_dim
        if emb_dim > len(spk_dict):
            for i in range(emb_dim - len(spk_dict)):
                spk_dict[f"spk{i}"] = len(spk_dict)
        if emb_dim < len(spk_dict):
            for i in range(len(spk_dict) - emb_dim):
                spk_dict.popitem()
        self.speaker = ','.join(spk_dict.keys())
        config_template['model']['ssl_dim'] = config_template["model"]["filter_channels"] = config_template["model"][
            "gin_channels"] = model_dim
        config_template['model']['n_speakers'] = diff_config_template['model']['n_spk'] = emb_dim
        config_template['spk'] = diff_config_template['spk'] = spk_dict
        encoder = [k for k, v in MODEL_TYPE.items() if v == model_dim]
        if orig_config['model']['speech_encoder'] in encoder:
            config_template['model']['speech_encoder'] = orig_config['model']['speech_encoder']
        else:
            raise Exception("Config is not compatible with the model")

        # if (diff_model != "no_diff" or
        if not os.path.exists(diff_model):
            diff_model = os.path.join(diff_read_dir, diff_model)
        # os.path.exists(os.path.join(diff_read_dir, diff_model))
        # ):
        if os.path.exists(diff_model):
            _diff = torch.load(diff_model, map_location='cpu')
            _, diff_dim = _diff["model"].get("unit_embed.weight", torch.empty(0, 0)).size()
            if diff_dim == 256:
                diff_config_template['data']['encoder'] = 'hubertsoft'
                diff_config_template['data']['encoder_out_channels'] = 256
            elif diff_dim == 768:
                diff_config_template['data']['encoder'] = 'vec768l12'
                diff_config_template['data']['encoder_out_channels'] = 768
            elif diff_dim == 1024:
                diff_config_template['data']['encoder'] = 'whisper-ppg'
                diff_config_template['data']['encoder_out_channels'] = 1024

        with open("release_packs/install.txt", 'w') as f:
            f.write(str(self.file_list) + '#' + str(self.speaker))

        _template.save(config_template)
        _d_template.save(diff_config_template)

    def unpack(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall("release_packs")

    def formatted_install(self, install_txt):
        with open(install_txt, 'r') as f:
            content = f.read()
        file_list, speaker = content.split('#')
        self.speaker = speaker
        file_list = ast.literal_eval(file_list)
        self.file_list = file_list
        for _, target_path in self.file_list:
            if target_path != "install.txt" and target_path != "":
                shutil.move(os.path.join("release_packs", target_path), target_path)
        self.remove_temp("release_packs")
        return self.speaker

    def pack(self):
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, target_path in self.file_list:
                if os.path.isfile(file_path):
                    zipf.write(file_path, arcname=target_path)


def release_packing(model_path, model_config, speaker, diff_path, cluster_path):
    # model_path = diff_path = cluster_path = ""
    model_to_pack = os.path.basename(model_path)
    model_name = os.path.splitext(model_to_pack)[0]

    diff_to_pack = os.path.basename(diff_path)
    diff_name = os.path.splitext(diff_to_pack)[0]
    cluster_to_pack = os.path.basename(cluster_path)

    # diff_basename = os.path.splitext(diff_to_pack)[0]
    if model_to_pack == "" or model_config == "" or speaker == "":
        return "存在必选项为空，请检查后重试"
    released_pack = ReleasePacker(speaker, model_path)
    released_pack.remove_temp("release_packs")
    # model_path = os.path.join("logs/44k", model_to_pack)
    if os.stat(model_path).st_size > 300000000:
        print("Model is too large, removing optimizer. =>", model_to_pack)
        output_model = os.path.join("release_packs", model_to_pack)
        removeOptimizer(model_config, model_path, False, output_model)
        model_path = output_model

    # if diff_to_pack != "no_diff":
    #     diff_path = os.path.join(diff_read_dir, diff_to_pack)

    # if c != "no_cluster":
    #     cluster_path = os.path.join(ckpt_read_dir, cluster_to_pack)
    shutil.copyfile("configs_template/config_template.json", "release_packs/config_template.json")
    shutil.copyfile("configs_template/diffusion_template.yaml", "release_packs/diffusion_template.yaml")
    files_to_pack = [
        (model_path, f"models/{model_to_pack}"),
        (f"release_packs/{model_name}.json", f"models/{model_name}.json"),
        ("release_packs/install.txt", "install.txt")
    ]
    if os.path.exists(diff_path):
        files_to_pack.extend(
            [(diff_path, f"models/diffusion/{diff_to_pack}"),
             (f"release_packs/{diff_name}.yaml", f"models/{diff_name}.yaml"), ])

    if os.path.exists(cluster_path):
        files_to_pack.append((cluster_path, f"models/{cluster_to_pack}"))
    # if
    # (diff_path, f"models/diffusion/{diff_to_pack}") if diff_to_pack != "no_diff" else ("", ""),
    # (cluster_path, f"models/{cluster_to_pack}") if cluster_to_pack != "no_cluster" else ("", ""),
    # (f"release_packs/{diff_basename}.yaml", f"models/{diff_basename}.yaml") if diff_to_pack != "no_diff" else (
    # "", ""),
    released_pack.add_file(files_to_pack)
    released_pack.generate_config(diff_to_pack, model_config)
    os.rename("release_packs/config_template.json", f"release_packs/{model_name}.json")
    os.rename("release_packs/diffusion_template.yaml", f"release_packs/{diff_name}.yaml")
    print("generate_config over")
    print("Packing...")
    released_pack.pack()
    to_remove = [file for file in os.listdir("release_packs") if not file.endswith(".zip")]
    print("remove temp files")
    for file in to_remove:
        os.remove(os.path.join("release_packs", file))
    print("Packing over. =>", released_pack.output_path)
    return "打包成功, 请在release_packs目录下查看"


if __name__ == '__main__':
    release_packing(
        "logs/lain3/G_3200.pth",
        "logs/lain3/config.json",
        "lain",
        "logs/lain3/diffusion/model_2000.pt",
        "logs/lain3/kmeans_10000.pt")
