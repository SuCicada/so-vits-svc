import os
from typing import NamedTuple, Dict


class ModelsInfo(NamedTuple):
    hf_repo: str
    remote_dir: str
    model_pth: str
    config_json: str
    kmeans_pt: str
    diffusion_config_yaml: str
    diffusion_model_pt: str

    def get_models_list(self):
        return [self.model_pth, self.config_json, self.kmeans_pt, self.diffusion_config_yaml, self.diffusion_model_pt]

    def to_svc_config(self, local_dir):
        return {
            "model_path": os.path.join(local_dir, self.model_pth),
            "config_path": os.path.join(local_dir, self.config_json),
            "cluster_model_path": os.path.join(local_dir, self.kmeans_pt),
            "diff_model_path": os.path.join(local_dir, self.diffusion_model_pt),
            "diff_config_path": os.path.join(local_dir, self.diffusion_config_yaml),
        }


_modelsInfoRegister: Dict[str, ModelsInfo] = {}


def getModelsInfo(name) -> ModelsInfo:
    return _modelsInfoRegister[name]


def _registerModelsInfo(name, info: ModelsInfo):
    _modelsInfoRegister[name] = info


LainV1 = ModelsInfo(
    hf_repo="SuCicada/Lain-so-vits-svc-4.1",
    remote_dir="models",
    model_pth="G_2400_infer.pth",
    config_json="config.json",
    kmeans_pt="kmeans_10000.pt",
    diffusion_config_yaml="diffusion/config.yaml",
    diffusion_model_pt="diffusion/model_12000.pt",
)
_registerModelsInfo("LainV1", LainV1)

LainV2 = ModelsInfo(
    hf_repo="SuCicada/Lain-so-vits-svc-4.1-v2",
    remote_dir="202309/models",
    model_pth="G_3200_compressed.pth",
    config_json="config.json",
    kmeans_pt="kmeans_10000.pt",
    diffusion_config_yaml="diffusion/config.yaml",
    diffusion_model_pt="diffusion/model_2000.pt",
)
_registerModelsInfo("LainV2", LainV2)
