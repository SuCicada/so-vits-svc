import onnx
onnx_path = "/Users/peng/PROGRAM/GitHub/so-vits-svc/checkpoints/lain/model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
import onnxruntime as ort
import numpy as np
# x, y = test_data[0][0], test_data[0][1]
# ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
# outputs = ort_sess.run(None, {'input': x.numpy()})
#
# # Print Result
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')
ort_sess = ort.InferenceSession(onnx_path)
outputs = ort_sess.run(None, {'input': text.numpy(),
                              'offsets':  torch.tensor([0]).numpy()})
# Print Result
result = outputs[0].argmax(axis=1)+1
print("This is a %s news" %ag_news_label[result[0]])
