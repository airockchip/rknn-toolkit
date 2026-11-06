from PIL import Image
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
import cv2
import sys
import argparse
import os
import time
import urllib
import traceback
import platform

from rknn.api import RKNN

# Needed to show segmentation colormap labels
import get_dataset_colormap

_HELP_TEXT = """
Usage Examples:
python test.py
python test.py --not-pass-through
python test.py --use-rknn
python test.py --target rk1808
python test.py --target rv1126 --device-id c3d9b8674f4b94f6
"""

INPUT_SIZE = 513
TEST_IMAGE = './bike_boy.jpg'
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)

def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)

def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')

def run(image, inference_result):
    """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    b = inference_result
    b.shape = 65 * 65, 21    # ResizeBilinear_2
    b = np.transpose(b)
    seg_img = np.argmax(b, axis=-2)
    seg_img = np.reshape(seg_img, (65, 65))    # ResizeBilinear_2

    return resized_image, seg_img

def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)

    # plt.show()
    plt.savefig('seg_results.jpg')

def deeplabv3_post_process(img, inference_result):
    origin_im = Image.open(img)
    print('--> Post process with image %s...' % img)
    resized_im, seg_map = run(origin_im, inference_result)

    vis_segmentation(resized_im, seg_map)
    print('done')

def get_args():
    """
    parse commandline
    """
    parser = argparse.ArgumentParser(description="RKNN Toolkit pass_through demo.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("--not-pass-through", action="store_true", default=False, help="Do not pass through input data.")
    parser.add_argument("--use-rknn", action="store_true", default=False, help="Use RKNN model directly.")
    parser.add_argument("--target", default='rv1126', help = "Target device, default is 'rv1126', other target device can be: 'rk1806', 'rk1808', 'rk3399pro', 'rv1109', 'rv1126'")
    parser.add_argument("--device-id", default=None,
                        help="Device id of target device, if multiple devices connected, '--target' and this parameter must be set.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    not_pass_through = args.not_pass_through
    use_rknn = args.use_rknn
    target = args.target
    device_id = args.device_id

    PB_MODEL = './deeplab-v3-plus-mobilenet-v2.pb'
    RKNN_MODEL = './deeplab-v3-plus-mobilenet-v2.rknn'

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(PB_MODEL):
        print('--> Download {}'.format(PB_MODEL))
        url = 'https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb'
        download_file = PB_MODEL
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
        print('done')

    if not use_rknn:
        print('--> Convert {} to {}'.format(PB_MODEL, RKNN_MODEL))
        # Set config refer to pass_througn value.
        rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                    std_values=[[127.5, 127.5, 127.5]],
                    reorder_channel='0 1 2',
                    target_platform=[target])

        # Load tensorflow model
        print('--> Loading model')
        ret = rknn.load_tensorflow(tf_pb=PB_MODEL,
                                   inputs=['sub_7'],
                                   outputs=['ResizeBilinear_2'],
                                   input_size_list=[[513, 513, 3]])
        if ret != 0:
            print('Load {} failed'.format(PB_MODEL))
            rknn.release()
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile=False)
        if ret != 0:
            print('Build rknn model failed')
            rknn.release()
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export rknn model failed')
            rknn.release()
            exit(ret)
        print('done')
    else:
        print('--> Load RKNN model')
        ret = rknn.load_rknn(RKNN_MODEL)
        if ret < 0:
            print('Load RKNN model failed.')
            rknn.release()
            exit(ret)
        print('done')

    # Init RKNN runtime
    print('--> init runtime')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret < 0:
        print('Init runtime failed')
        rknn.release()
        exit(ret)
    print('done')

    # Inference
    print('--> inference')

    # Get input data
    img = cv2.imread(TEST_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # preprocess if pass_through is True
    if not_pass_through:
        print('Pass normal data to runtime, runtime will do preprocess and quantize.')
        inputs_pass_through = [0]
    else:
        print('Pass through input data to model directly.')
        img = (img.astype(np.float32)-127.5)/127.5
        # quantize input
        scale = 0.007843137718737125
        zp = 127
        img = ((img / scale) + zp).astype(np.uint8)
        inputs_pass_through=[1]

    outputs = rknn.inference(inputs=[img], inputs_pass_through=inputs_pass_through)
    for idx, out in enumerate(outputs):
        if not_pass_through:
            np.save('out{}_norm.npy'.format(idx), out)
        else:
            np.save('out{}_pass.npy'.format(idx), out)
    print('done')

    rknn.release()

    deeplabv3_post_process(img=TEST_IMAGE, inference_result=outputs[0])

    exit(0)
