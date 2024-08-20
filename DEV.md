init
```bash
conda create --name so-vits-svc python=3.8 -y
. ./activate.sh
```


reference:
- [Makefile](Makefile)
- [sumake](https://github.com/SuCicada/sumake)

## inference
```bash
sumake lain_download
sumake lain_gradio_run

# only download models (include by lain_download)
python tools/infer/download_models.py
```

## train
```bash
# release_packing for app.py
python tools/webui/release_packing.py

# upload
pyhton tools/upload_huggingface.py

python compress_model.py -c models/config.json -i  tmp/G_3200.pth  -o tmp/G_3200_compressed.pth
```

## notice
- MemoryError: Cannot allocate write+execute memory for ffi.callback(). You might be running on a system that prevents this. For more information, see https://cffi.readthedocs.io/en/latest/using.html#callbacks
mac 上会这样， 用 python 3.10
