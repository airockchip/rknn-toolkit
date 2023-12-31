import sys
from rknn.api import RKNN

ONNX_MODEL = 'shufflenetv2_x1.onnx'

if __name__ == '__main__':
    # Default target platform
    target = 'rv1126'

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) > 2:
        print('Too much arguments')
        print('Usage: python {} [target]'.format(sys.argv[0]))
        print('Such as: python {} rv1126'.format(
            sys.argv[0]))
        exit(-1)

    # Create RKNN object
    rknn = RKNN()
    
    # model config
    print('--> Config model')
    rknn.config(mean_values=[[123.68, 116.28, 103.53]],
                std_values=[[57.38, 57.38, 57.38]],
                reorder_channel='0 1 2',
                target_platform=[target])
    print('done')

    # Load onnx model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Hybrid quantization step1
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # Tips
    print('Please modify shufflenetv2_x1.quantization.cfg!')
    print('==================================================================================================')
    print('Modify method:')
    print('Add {layer_name}: {quantized_dtype} to dict of customized_quantize_layers')
    print('If no layer changed, please set {} as empty directory for customized_quantize_layers')
    print('==================================================================================================')
    print('Notes:')
    print('1. The layer_name comes from quantize_parameters, please strip \'@\' and \':xxx\';')
    print('   If layer_name contains special characters, please quote the layer name.')
    print('2. Support quantized_type: asymmetric_affine-u8, dynamic_fixed_point-i8, dynamic_fixed_point-i16, float32.')
    print('3. Please fill in according to the grammatical rules of yaml.')
    print('4. For this model, RKNN Toolkit has provided the corresponding configuration, please directly proceed to step2.')
    print('==================================================================================================')

    rknn.release()

