# Example for quantized ONNX shufflenet-v2


## Model Source
Shufflenet-v2_quant.onnx was generated with onnxruntime_v1.5.1 version via the "onnxruntime_quant_e2e_user_example.py" script.

The original float model is available at [https://github.com/onnx/models/tree/main/vision/classification/shufflenet](https://github.com/onnx/models/tree/main/vision/classification/shufflenet) named ShuffleNet-v2-fp32.

The quantization script was based on [https://github.com/microsoft/onnxruntime/blob/rel-1.5.1/onnxruntime/python/tools/quantization/E2E_example_model/e2e_user_example.py](https://github.com/microsoft/onnxruntime/blob/rel-1.5.1/onnxruntime/python/tools/quantization/E2E_example_model/e2e_user_example.py) We adjusted some std/mean value settings and included the script in this demo.

Note: 

- The quantized ONNX model was generated using onnxruntime_v1.5.1. Using other versions of onnxruntime may result in errors in the demo.
- [https://github.com/onnx/models/tree/main/vision/classification/shufflenet](https://github.com/onnx/models/tree/main/vision/classification/shufflenet) has already provide quantized shufflenet. It's per-channel quantized, which is not support due to hardware limitation.




## Usage for the script

*Usage:*
```
python test.py [target] [device_id]
```
*Parameter Description:*
- target: target platform. Optional parameter, the default value is `rv1126`, you can fill in `rk1806`, `rk1808`, `rk3399pro`, `rv1109`, `rv1126`.
- device_id: Device ID, when multiple devices are connected, this parameter is used to distinguish them. Optional parameter, default value is None.

If the target device is `RV1109` or `RV1126`, you can directly execute the following command to run the example:
```
python test.py
```
If the target device is RK1806, RK1808 or RK3399Pro, you can execute the following command to run the example:
```
python test.py rk1808
```
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example::
```
python test.py rv1126 c3d9b8674f4b94f6
```


## Expected results

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
--> Running rknn model
model: shufflenet-v2_quant
-----TOP 5-----
[155]: score=0.988016 class="n02086240 Shih-Tzu"
[154]: score=0.010231 class="n02086079 Pekinese, Pekingese, Peke"
[364]: score=0.000309 class="n02457408 three-toed sloth, ai, Bradypus tridactylus"
[960]: score=0.000173 class="n07836838 chocolate sauce, chocolate syrup"
[879]: score=0.000133 class="n04507155 umbrella"

--> Running onnx model
model: shufflenet-v2_quant
-----TOP 5-----
[155]: score=0.987892 class="n02086240 Shih-Tzu"
[154]: score=0.009038 class="n02086079 Pekinese, Pekingese, Peke"
[960]: score=0.000527 class="n07836838 chocolate sauce, chocolate syrup"
[879]: score=0.000268 class="n04507155 umbrella"
[364]: score=0.000256 class="n02457408 three-toed sloth, ai, Bradypus tridactylus"
```

1. The label index with the highest score is 155, the corresponding label is `Shih-Tzu`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
