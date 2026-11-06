# Example for MXNet resnext50_32x4d


## Model Source
The model used in this example come from the GluonCV project:
https://cv.gluon.ai/model_zoo/classification.html


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
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example:
```
python test.py rv1126 c3d9b8674f4b94f6
```


## Expected results

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
model: resnext50
-----TOP 5-----
[812]: score=0.999962 class="n04266014 space shuttle"
[460]: score=0.000001 class="n02894605 breakwater, groin, groyne, mole, bulwark, seawall, jetty"
[525]: score=0.000001 class="n03160309 dam, dike, dyke"
[061]: score=0.000001 class="n01742172 boa constrictor, Constrictor constrictor"
[628]: score=0.000000 class="n03673027 liner, ocean liner"
```

1. The label index with the highest score is 812, the corresponding label is space shuttle.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
