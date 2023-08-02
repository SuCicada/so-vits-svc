#ifneq ($(conda),false)
.PHONY: lain
CONDA_ENV=so-vits-svc
lain_dir=models
test:
	@echo $(lain_dir)
lain_download:
	#$(call wget_if_not_exist, \
#			$(lain_dir)/checkpoint_best_legacy_500.pt, \
#			https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
	$(call wget_if_not_exist, \
		pretrain/checkpoint_best_legacy_500.pt  ,\
		https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt )

	$(call wget_if_not_exist, \
		pretrain/nsf_hifigan/nsf_hifigan_20221211.zip  ,\
		https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip )
	unzip -o pretrain/nsf_hifigan/nsf_hifigan_20221211.zip -d pretrain

#$(call wget_if_not_exist, \
#			$(lain_dir)/config.json, \
#			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/config.json)
#	$(call wget_if_not_exist, \
#			$(lain_dir)/G_256800_infer.pth, \
#			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/G_256800_infer.pth)
#	$(call wget_if_not_exist, \
#			$(lain_dir)/kmeans_10000.pt, \
#			https://huggingface.co/SuCicada/Lain-so-vits-svc-4.0/resolve/main/kmeans_10000.pt)

_lain_gradio_cmd = $(conda_run) python tools/lain_gradio.py \
      --model_path $(lain_dir)/G_2400_infer.pth \
      --config_path $(lain_dir)/config.json \
      --diff_model_path $(lain_dir)/model_12000.pt \
      --diff_config_path $(lain_dir)/config.yaml \
      --cluster_model_path $(lain_dir)/kmeans_10000.pt \

lain_gradio_run:
	$(_lain_gradio_cmd) --port 17861
lain_gradio_debug:
	$(_lain_gradio_cmd) --debug

lain_server:
	$(conda_run) python tools/server.py
