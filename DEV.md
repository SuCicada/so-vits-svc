todo ......


reference:
- [Makefile](Makefile)
- [sumake](https://github.com/SuCicada/sumake)

## install
```bash
sumake lain_download
sumake lain_gradio_run
```

## upload
```bash
pyhton tools/upload_huggingface.py
#python tools/infer/download_models.py
```    

## notice
- MemoryError: Cannot allocate write+execute memory for ffi.callback(). You might be running on a system that prevents this. For more information, see https://cffi.readthedocs.io/en/latest/using.html#callbacks
mac 上会这样， 用 python 3.10
