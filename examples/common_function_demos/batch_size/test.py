import platform
import sys
import numpy as np
import cv2
from rknn.api import RKNN

RKNN_BATCH_SIZE = 4
CLASS_LABEL_PATH = './labels.txt'


def show_top5(outputs, input_path):
    with open(CLASS_LABEL_PATH, 'r') as f:
        labels = [l.rstrip() for l in f]

    output = outputs[0].reshape(-1)
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'TOP 5 for input: {}\n'.format(input_path)
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:03d}]: score={:.6f} class="{}"\n'.format(
                index, value, labels[index])
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
    rknn = RKNN(verbose=False)

    # Set model config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                std_values=[[127.5, 127.5, 127.5]],
                reorder_channel='0 1 2',
                quantized_dtype='asymmetric_quantized-u8',
                target_platform=[target])
    print('done')

    # Load TFLite model
    print('--> Loading model')
    ret = rknn.load_tflite(model='../../tflite/mobilenet_v1/mobilenet_v1.tflite')
    if ret != 0:
        print('Load mobilenet_v1 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt', rknn_batch_size=RKNN_BATCH_SIZE)
    if ret != 0:
        print('Build mobilenet_v1 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./mobilenet_v1.rknn')
    if ret != 0:
        print('Export mobilenet_v1.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs

    # get input paths from dataset.txt
    with open('dataset.txt', 'r') as f:
        paths = f.readlines()
    img_list = []
    assert len(paths) == RKNN_BATCH_SIZE
    for i in range(RKNN_BATCH_SIZE):
        img = cv2.imread(paths[i].strip())
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img_list.append(img)

    imgs = np.concatenate(tuple(img_list), axis=0)

    # init runtime environment
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
    outputs = rknn.inference(inputs=[imgs])
    # outputs[0] shape: 4, 1001
    outputs = outputs[0].reshape((-1, 1001))
    print('model: mobilenet_v1')
    for idx, out in enumerate(outputs):
        show_top5([out], paths[idx].strip())
    print('done')

    rknn.release()
