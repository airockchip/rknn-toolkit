import platform
import sys
import torch
import cv2
import numpy as np
from rknn.api import RKNN
import torchvision

mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

mean = [m*255 for m in mean]
std = [s*255 for s in std]

model_path = './quantized_mobilenet.pt'

def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)

def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def prepare_model():
    mobilenet_v2 = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

    img = cv2.imread('dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean[i])/std[i]

    input = torch.from_numpy(img).reshape(1,*img.shape)
    input = input.permute(0,3,1,2)

    traced_model = torch.jit.trace(mobilenet_v2,input)
    torch.jit.save(traced_model, model_path)


def main():

    print('*'*20)
    print("NOTE:")
    print("    To run this demo, it's recommanded to use PyTorch1.9.0 and Torchvision0.10.0")
    print('*'*20)

    prepare_model()

    # Default target and device_id
    target = 'rv1126'
    device_id = None

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 3:
        target = sys.argv[1]
        device_id = sys.argv[2]
        print('Set target: {}, device_id: {}'.format(target, device_id))
    elif len(sys.argv) > 3:
        print('Too much arguments')
        print('Usage: python {} [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    rknn = RKNN(verbose=False, verbose_file='./verbose_log.txt')

    # pre-process config
    print('--> Set config model')
    rknn.config(
                quantize_input_node=True,
                merge_dequant_layer_and_output_node=True,
                target_platform=[target],
                optimization_level=3,
                )
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model_path, input_size_list=[[3, 224, 224]])
    if ret != 0:
        print('Load Pytorch model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='dataset.txt')
    if ret != 0:
        print('Build model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('quantized_mobilenet.rknn')
    if ret != 0:
        print('Export quantized_mobilenet.rknn failed!')
        rknn.release()
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean[i])/std[i]
    
    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('rk_result:')
    show_outputs(softmax(np.array(outputs[0][0])))
    #! NOTE:
    # rknn_model got 70.0 accuracy on imagenet_1000
    # pt_i8 got 61.5 accuracy on imagenet_1000
    # pt_i8 got 71.658 accuracy refer to torchvsion discription 
    rknn.release()

    torch.backends.quantized.engine = 'qnnpack'
    pt_model = torch.jit.load(model_path).eval()
    img = cv2.imread('dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean[i])/std[i]

    input = torch.from_numpy(img).reshape(1,*img.shape)
    input = input.permute(0,3,1,2)

    pt_result = pt_model(input)
    pt_result = torch.dequantize(pt_result)
    print('pt_result:')
    show_outputs(softmax(pt_result[0].numpy()))


if __name__ == '__main__':
    main()
