# Example for multi-batch


## Model Source
The model used in this example come from the TensorFlow Lite offical model zoo:
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md


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

This example will print the TOP5 labels and corresponding scores of the test images classification results. For example, the inference results of this example are as follows:
```
model: mobilenet_v1
TOP 5 for input: Shih-Tzu_224.jpg
[156]: score=0.853027 class="Shih-Tzu"
[155]: score=0.094116 class="Pekinese, Pekingese, Peke"
[205]: score=0.012024 class="Lhasa, Lhasa apso"
[284]: score=0.006683 class="Persian cat"
[194]: score=0.002062 class="Australian terrier"

TOP 5 for input: space_shuttle_224.jpg
[813]: score=0.999512 class="space shuttle"
[405]: score=0.000230 class="airliner"
[868]: score=0.000061 class="trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi"
[ -1]: score=0.0
[ -1]: score=0.0

TOP 5 for input: monkey.jpg
[371]: score=0.998047 class="guenon, guenon monkey"
[372]: score=0.001553 class="patas, hussar monkey, Erythrocebus patas"
[374]: score=0.000128 class="macaque"
[381]: score=0.000082 class="titi, titi monkey"
[ -1]: score=0.0

TOP 5 for input: bee_eater.jpg
[093]: score=0.756348 class="bee eater"
[321]: score=0.072021 class="damselfly"
[320]: score=0.062195 class="dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk"
[396]: score=0.014305 class="gar, garfish, garpike, billfish, Lepisosteus osseus"
[015]: score=0.010666 class="indigo bunting, indigo finch, indigo bird, Passerina cyanea"
```

1. The label used in this model contains background, so its index will be 1 more than other examples.
2. Different platforms, different versions of tools and drivers may have slightly different results.


## Notes

1. the inputs data need to be merged together using np.concatenate.
```
e.g:
    img = np.concatenate((img1, img2, img3, img4), axis=0)
```
