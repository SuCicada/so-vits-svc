#ifneq ($(conda),false)
.PHONY: lain
CONDA_ENV=so-vits-svc
lain_dir=models
test:
	@echo $(lain_dir)
lain_download:
	$(call wget_if_not_exist, \
			$(lain_dir)/checkpoint_best_legacy_500.pt, \
			https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
	$(call wget_if_not_exist, \
			$(lain_dir)/config.json, \
			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/config.json)
	$(call wget_if_not_exist, \
			$(lain_dir)/G_256800_infer.pth, \
			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/G_256800_infer.pth)
	$(call wget_if_not_exist, \
			$(lain_dir)/kmeans_10000.pt, \
			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/kmeans_10000.pt)

_lain_gradio_cmd = $(conda_run) python tools/lain_gradio.py \
      --model_path $(lain_dir)/G_256800_infer.pth \
      --config_path $(lain_dir)/config.json \
      --cluster_model_path $(lain_dir)/kmeans_10000.pt \
      --hubert_model_path $(lain_dir)/checkpoint_best_legacy_500.pt
PORT = 17861
lain_gradio_run:
	$(_lain_gradio_cmd) --port $(PORT)
lain_gradio_debug:
	$(_lain_gradio_cmd) --debug

lain_server:
	$(conda_run) python tools/server.py
