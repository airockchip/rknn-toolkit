import platform
import sys
import numpy as np
import cv2
from rknn.api import RKNN

CLASS_LABEL_PATH='./synset_words.txt'

def show_outputs(outputs):
    with open(CLASS_LABEL_PATH, 'r') as f:
        labels = [l.rstrip() for l in f]

    output = outputs[0].reshape(-1)
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'model: mobilenet_v2\n-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:03d}]: score={:.6f} class="{}"\n'.format(index, value, labels[index])
        else:
            topi = '[ -1]: score=0.0\n'
        top5_str += topi
    print(top5_str)


if __name__ == '__main__':
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

    # Create RKNN object
    rknn = RKNN()
    
    # Set model config
    print('--> Config model')
    rknn.config(mean_values=[[103.94, 116.78, 123.68]],
                std_values=[[58.82, 58.82, 58.82]],
                reorder_channel='2 1 0',
                target_platform=[target])
    print('done')

    # Load model (from https://github.com/shicai/MobileNet-Caffe)
    print('--> Loading model')
    ret = rknn.load_caffe(model='./mobilenet_v2.prototxt',
                          proto='caffe',
                          blobs='./mobilenet_v2.caffemodel')
    if ret != 0:
        print('Load mobilenet_v2 failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build mobilenet_v2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./mobilenet_v2.rknn')
    if ret != 0:
        print('Export mobilenet_v2.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./goldfish_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    show_outputs(outputs)
    print('done')

    # Evaluate model performance
    print('--> Evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.release()

