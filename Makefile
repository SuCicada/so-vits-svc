#ifneq ($(conda),false)
ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: lain
CONDA_ENV=so-vits-svc
lain_dir=models

install:
	pip install pip==23.1.2
	pip install -r requirements_infer.txt
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
#		pretrain/rmvpe.pt  ,\
#		https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt )
	$(call wget_if_not_exist, \
		pretrain/rmvpe.zip  ,\
		https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip )
	unzip -o pretrain/rmvpe.zip -d pretrain && mv pretrain/model.pt pretrain/rmvpe.pt

# models
	python tools/infer/download_models.py
#	$(call wget_if_not_exist, \
#		models/G_2400_infer.pth  ,\
#		https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1/resolve/main/G_2400_infer.pth )
#	$(call wget_if_not_exist, \
#		models/config.json  ,\
#		https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1/resolve/main/config.json )
#	$(call wget_if_not_exist, \
#		models/kmeans_10000.pt  ,\
#		https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1/resolve/main/kmeans_10000.pt )
#	$(call wget_if_not_exist, \
#		models/diffusion/config.yaml  ,\
#		https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1/resolve/main/diffusion/config.yaml )
#	$(call wget_if_not_exist, \
#		models/diffusion/model_12000.pt  ,\
#		https://huggingface.co/SuCicada/Lain-so-vits-svc-4.1/resolve/main/diffusion/model_12000.pt )

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
       --model LainV2
#      --model_path $(lain_dir)/G_2400_infer.pth \
#      --config_path $(lain_dir)/config.json \
#      --diff_model_path $(lain_dir)/diffusion/model_12000.pt \
#      --diff_config_path $(lain_dir)/diffusion/config.yaml \
#      --cluster_model_path $(lain_dir)/kmeans_10000.pt \
#

define sumake_run
	if [ -z "${SUMAKE}" ]; then \
		sumake $@   ;\
	else \
		$1  ;\
	fi
endef

lain_gradio_run:
	$(call sumake_run, $(_lain_gradio_cmd) --port 17861 $(args) )  #

lain_gradio_debug:
	$(call sumake_run, $(_lain_gradio_cmd) --debug --port 17861 )

test_server:
	$(call sumake_run, $(conda_run) python tools/test_server.py )

lain_server:
	$(conda_run) python tools/server.py

requirements-lock:
	$(conda_run) pip freeze > requirements-infer-lock.txt

upload:
	$(call upload, .env, $(DEPLOY_PATH)/)

docker-run-local:
	cd deploy/local && \
		docker-compose down && \
		docker-compose up -d
