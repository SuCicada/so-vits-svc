import torch
import sys
import os
import torch.multiprocessing as multiprocessing
if len(sys.argv) < 2:
    print("Usage: python3 transform_infer.py <pth_file> <output_file>")
    exit()
pth_file = sys.argv[1]
if len(sys.argv) > 2:
    output_file = sys.argv[2]
else:
    output_file = os.path.splitext(pth_file)[0] + "_infer.pth"
# pth_file = "test.pth"
# output_file = "infer.pth"
# if torch.cuda.is_available() and torch.cuda.is_mps_supported():
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(pth_file, map_location=torch.device(device))
print("read origin file :", output_file)
print("origin file size :", os.path.getsize(pth_file))

torch.save({
    "model": model["model"],
    "learning_rate": model["learning_rate"],
    "iteration": model["iteration"],
}, output_file)
print("inference file size :", os.path.getsize(output_file))
print("save inference file :", output_file)

from torch.quantization import quantize_dynamic
from pympler import asizeof
from pympler import tracker
# pth_file= "logs/32k/G_50000.pth"
# print(os.path.getsize(pth_file))
# print(model.keys())
# print(asizeof.asizeof(model))
# print(asizeof.asizeof(model["model"]))
# print(type(model["model"]))
# print(model["iteration"])
# print(asizeof.asizeof(model["optimizer"]))
# quantized_model = quantize_dynamic(model["model"], {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
# output="logs/32k/quantized_model.pth"
