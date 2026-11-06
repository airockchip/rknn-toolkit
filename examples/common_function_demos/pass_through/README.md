# Example of passing through input data


## Model Source
The model used in this example come from the miai-models project, download link:
https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb


## Usage for the script

*Usage:*
```
python test.py --not-pass-through --use-rknn --target --device-id
```
*Parameter Description:*
- --not-pass-through: whether to pass through input data. Optional parameter. If this parameter is not set, this script will pass through input data to NPU directly, otherwise, the rknn runtime will pre-process and quantized input data, then pass to NPU.
- use-rknn: whether to use RKNN model directly. Optional paramter. If this parameter is set, the script will load RKNN model directly, please make sure the RKNN model exists.
- target: target platform. Optional parameter, the default value is `rv1126`, you can fill in `rk1806`, `rk1808`, `rk3399pro`, `rv1109`, `rv1126`.
- device_id: Device ID, when multiple devices are connected, this parameter is used to distinguish them. Optional parameter, default value is None.

If the target device is `RV1109` or `RV1126`, you can directly execute the following command to run the example:
```
python test.py
```
If you do not want to pass through input data, you can execute the following command to run the example:
```
python test.py --not-pass-through
```
If you have already done model conversion and want to use the RKNN model directly for testing, you can execute the following command to run the example:
```
python test.py --use-rknn
```
If the target device is RK1806, RK1808 or RK3399Pro, you can execute the following command to run the example:
```
python test.py --target rk1808
```
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example:
```
python test.py --target rv1126 --device-id c3d9b8674f4b94f6
```

## Expected results

The test result should be similar to picutre `ref_seg_results.jpg`.

*Note: Different platforms, different versions of tools and drivers may have slightly different results.*
