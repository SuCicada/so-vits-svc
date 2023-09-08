from typing import NamedTuple


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


LainV1 = ModelsInfo(
    hf_repo="SuCicada/Lain-so-vits-svc-4.1",
    remote_dir="models",
    model_pth="G_2400_infer.pth",
    config_json="config.json",
    kmeans_pt="kmeans_10000.pt",
    diffusion_config_yaml="diffusion/config.yaml",
    diffusion_model_pt="diffusion/model_12000.pt",
)

LainV2 = ModelsInfo(
    hf_repo="SuCicada/Lain-so-vits-svc-4.1-v2",
    remote_dir="202309/models",
    model_pth="G_3200_compressed.pth",
    config_json="config.json",
    kmeans_pt="kmeans_10000.pt",
    diffusion_config_yaml="diffusion/config.yaml",
    diffusion_model_pt="diffusion/model_2000.pt",
)
