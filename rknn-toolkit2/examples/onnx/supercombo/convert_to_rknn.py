import argparse
import os
import urllib
import traceback
import time
import sys
from scipy.spatial.distance import cosine
import numpy as np
import cv2
from rknn.api import RKNN
import onnx
import itertools

ONNX_MODEL = 'supercombo.onnx'
RKNN_MODEL = 'supercombo.rknn'

#mean_values = [[111., 111., 111., 111., 144., 112., 111., 111., 111., 111., 114., 112.], [111., 111., 111., 111., 144., 112., 111., 111., 111., 111., 114., 112.]]
mean_values = None
#std_values = [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]
std_values = None

def attributeproto_fp16_to_fp32(attr):
  float32_list = np.frombuffer(attr.raw_data, dtype=np.float16)
  attr.data_type = 1
  attr.raw_data = float32_list.astype(np.float32).tobytes()

def convert_fp16_to_fp32(path):
  model = onnx.load(path)
  for i in model.graph.initializer:
    if i.data_type == 10:
      attributeproto_fp16_to_fp32(i)
  for i in itertools.chain(model.graph.input, model.graph.output):
    if i.type.tensor_type.elem_type == 10:
      i.type.tensor_type.elem_type = 1
  for i in model.graph.node:
    for a in i.attribute:
      if hasattr(a, 't'):
        if a.t.data_type == 10:
          attributeproto_fp16_to_fp32(a.t)
  return model.SerializeToString()

if __name__ == '__main__':
    perf_report = "-p" in sys.argv
    get_accuracy = "-a" in sys.argv

    model_data = convert_fp16_to_fp32(ONNX_MODEL)
    with open("/tmp/fp32.onnx", "wb") as f:
        f.write(model_data)

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588', optimization_level=0)
    print('done')

    # Load ONNX modeld
    print('--> Loading model')
    ret = rknn.load_onnx(model='/tmp/fp32.onnx')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Generate Cpp demo code
    print('--> Generate cpp demo')
    ret = rknn.codegen(output_path='./generated_cpp', overwrite=True)
    if ret != 0:
        print('Generate cpp demo failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=('rk3588' if perf_report else None), perf_debug=perf_report, eval_mem=perf_report)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    if perf_report:
        rknn.list_devices()
    
        # eval perf
        print('--> Eval perf')
        rknn.eval_perf()

        # eval mem
        print('--> Eval mem')
        rknn.eval_memory()

    if get_accuracy:
      # Accuracy eval
      onnx_output = np.load('onnx_output.npy')
      onnx_input = np.load('onnx_input.npy', allow_pickle=True)
      inputs_names = [v for v in onnx_input.item()]
      inputs = [onnx_input.item().get(name) for name in inputs_names]

      outputs = rknn.inference(inputs=inputs, inputs_pass_through=[0,0,0,0,0,0,0,0,0], data_format=['nchw', 'nchw', 'nchw', 'nchw', 'nchw', 'nchw', 'nchw', 'nchw', 'nchw'])
    
      # Calculate the MSE
      mse = (np.square(onnx_output - outputs)).mean(axis=None)
      print("MSE: " + str(mse))

      # Calculate the cosine similarity
      cosine_similarity = 1 - cosine(onnx_output[0][0], outputs[0][0])
      print("Cosine similarity: " + str(cosine_similarity))

    rknn.release()
