import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'dmonitoring_model.onnx'
RKNN_MODEL = 'dmonitoring_model.rknn'
DATASET = 'dataset.txt'
IMG_PATH = 'ecam.jpeg'
QUANTIZE_ON = False

IMG_SIZE = (1440, 960)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def post_process(res):
    print("face orientation 1: ")
    print(res[0][0][0:3])
    print("face position 1 (dx, dy):")
    face_pos = res[0][0][3:5] * 0.25
    print(res[0][0][3:5] * 0.25)
    print("normalised face 1: ")
    print(res[0][0][5:6])
    print("std dev 1: ")
    print(res[0][0][6:12])
    print("face prob: ")
    print(sigmoid(res[0][0][12:13]))
    print("eye position and size and their std: ")
    print(res[0][0][13:31])
    print("eye visible: ")
    print(sigmoid(res[0][0][31:32]))
    print("eye closed: ")
    print(sigmoid(res[0][0][32:33]))
    print("sunglasses: ")
    print(sigmoid(res[0][0][33:34]))
    print("face occluded: ")
    print(sigmoid(res[0][0][34:35]))
    print("wheel touch: ")
    print(sigmoid(res[0][0][35:36]))
    print("pay attention: ")
    print(sigmoid(res[0][0][36:37]))
    print("distracted probs (deprecated): ")
    print(sigmoid(res[0][0][37:39]))
    print("phone: ")
    print(sigmoid(res[0][0][39:30]))
    print("distracted probs new: ")
    print(sigmoid(res[0][0][41:42]))
    print("left hand drive prob: ")
    print(sigmoid(res[0][0][-2:-1]))
    return face_pos

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    #rknn.config(mean_values=[[128,128,128]], std_values=[[255,255,255]], target_platform="rk3588")
    rknn.config(target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON)
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

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs: Image input of [1, 1382400]
    img_ori = cv2.imread(IMG_PATH)
    #img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YUV_I420)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YUV)
    img_ori = img_ori[:, :, 0]
    img_ori = cv2.resize(img_ori, IMG_SIZE)

    img = np.reshape(img_ori.flatten(), (1, IMG_SIZE[0] * IMG_SIZE[1]))
    print("Input 0 has dimension: ")
    print(img.shape)

    img_flipped = cv2.flip(img_ori, 1)
    img_flipped = np.reshape(img_flipped.flatten(), (1, IMG_SIZE[0] * IMG_SIZE[1]))
    print("Input 0 flipped has dimension: ")
    print(img_flipped.shape)

    # Set inputs: Calib inputs of [1, 3]
    calib_inputs = np.expand_dims(np.zeros(3, dtype=np.float32), axis=0)
    print("Input 1 has dimension: ")
    print(calib_inputs.shape)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img, calib_inputs], data_format=None)
    outputs_flipped = rknn.inference(inputs=[img_flipped, calib_inputs], data_format=None)
   
    # Save the output
    np.save('result.npy', outputs)
    np.save('result_flipped.npy', outputs_flipped)

    # post process
    #face_pos = post_process(outputs)
    rknn.release()
