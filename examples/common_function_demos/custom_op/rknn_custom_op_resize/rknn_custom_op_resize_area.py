import cv2
import numpy as np
from rknn.api import RKNN
import sys

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

    rknn = RKNN(verbose=False)

    # RKNN model config
    print("--> Model config")
    rknn.config(target_platform=[target])
    print("done")

    # Register customer op.
    print("--> Register customer op.")
    rknn.register_op('./resize_area/ResizeArea.rknnop')
    print("done")

    # Load TensorFlow model with customer op.
    print("--> Load TF model.")
    ret = rknn.load_tensorflow(tf_pb='./resize_area_test.pb',
                               inputs=['input'],
                               outputs=['resize_area_0'],
                               input_size_list=[[32, 32, 3]])
    if ret != 0:
        print("Load TF model failed.")
        rknn.release()
        exit(-1)
    print("done")

    # Build RKNN model.
    print("--> Build RKNN model")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build RKNN model failed.")
        rknn.release()
        exit(ret)
    print("done")

    # rknn.export_rknn('./resize_area.rknn')

    # rknn.load_rknn('./resize_area.rknn')

    # Inference with RKNN.
    print("--> Init RKNN runtime")
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print("Init runtime failed.")
        rknn.release()
        exit(ret)
    print("done")

    img = cv2.imread('./dog_32x32.jpg')

    outs = rknn.inference(inputs=[img])

    # Show inference result.
    out_img = outs[0].astype('uint8')
    out_img = np.reshape(out_img, (64, 64, 3))
    cv2.imwrite('./out.jpg', out_img)

    rknn.release()
