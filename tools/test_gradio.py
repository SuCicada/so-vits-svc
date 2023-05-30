import sys

sys.argv = ["tools/lain_gradio.py",
            "--model_path", "models/G_256800_infer.pth",
            "--config_path", "models/config.json",
            "--cluster_model_path", "models/kmeans_10000.pt",
            "--hubert_model_path","models/checkpoint_best_legacy_500.pt"
            # "--debug",
            ]
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from tools import lain_gradio
lain_gradio.main()
