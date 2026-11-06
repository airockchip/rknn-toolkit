# Introduction
RKNN-Toolkit is a software development kit that provides users with model conversion, inference and performance evaluation on PC and Rockchip NPU platforms(RK1808/RK1806/RK3399Pro/RV1109/RV1126).  
RKNN-Toolkit-Lite provides Python programming interfaces for Rockchip NPU platform to help users deploy RKNN models and accelerate the implementation of AI applications.
Note:
- For the deployment of the RKNN model, please refer to:
  - RK1808/RK1806/RV1109/RV1126: https://github.com/airockchip/rknpu
  - RK3399Pro: https://github.com/airockchip/RK3399Pro_npu
- For RK3562/RK3566/RK3568/RK3576/RK3588/RV1103/RV1103B/RV1106/RV1106B, please refer to:
  - https://github.com/airockchip/rknn-toolkit2
# Download
From version 1.3.0, some wheel packages are larger than 100MB, can not be uploaded directly. So, you need to go to the releases page to download.
You can download from releases page: https://github.com/airockchip/rknn-toolkit/releases
- All wheel packages are in compressed file: [rknn-toolkit-v1.7.7-packages.tar.gz](https://github.com/airockchip/rknn-toolkit/releases/download/v1.7.7/rknn-toolkit-v1.7.7-packages.tar.gz "rknn-toolkit-v1.7.7-packages.tar.gz") or [rknn-toolkit-v1.7.7-packages.zip](https://github.com/airockchip/rknn-toolkit/releases/download/v1.7.7/rknn-toolkit-v1.7.7-packages.zip "rknn-toolkit-v1.7.7-packages.zip ")
- All examples, docs and platform-tools are in compressed file: [Source code(zip)](https://github.com/airockchip/rknn-toolkit/archive/v1.7.7.zip "Source code(zip)") or [Source code(tar.gz)](https://github.com/airockchip/rknn-toolkit/archive/v1.7.7.tar.gz "Source code(tar.gz)")
- You can also download all packages, docker image, examples, docs and platform-tools from: [RKNPU1_SDK](https://console.zbox.filez.com/l/t5dxIs" RKNPU1_SDK"), fetch code: rknn
- You can download RKNN Toolkit Lite packages and examples from [rknn-toolkit-lite](rknn-toolkit-lite)
# Checksums
## MD5
```
10d2be2a9acccbff3e2032afe1690bfe  rknn_toolkit-1.7.7-cp35-cp35m-linux_aarch64.whl
72e8409b550f4825fa191acfd105bf26  rknn_toolkit-1.7.7-cp36-cp36m-linux_x86_64.whl
61346b7c2a31f3d6560494409fb89104  rknn_toolkit-1.7.7-cp36-cp36m-macosx_10_15_x86_64.whl
78a705c24bd164d2a24549fd49c812f9  rknn_toolkit-1.7.7-cp36-cp36m-win_amd64.whl
8ea78f9cf4a115547e6d7c3ef15c11e3  rknn_toolkit-1.7.7-cp37-cp37m-linux_aarch64.whl
9f1318f7a1de10ae58e48c6bc3e40816  rknn_toolkit-1.7.7-cp37-cp37m-macosx_10_15_x86_64.whl
eeee336dc17f629c33094f5dc9c94567  rknn_toolkit-1.7.7-cp38-cp38-linux_x86_64.whl

699518a5792b66d5f7bdc627342c20a8  rknn-toolkit-v1.7.7-packages.tar.gz
ed5d9637fff4cbc3ce6d9f17e1b608d6  rknn-toolkit-v1.7.7-packages.zip

217b3caf09b13206ec1d749ff7bc9b87  rknn_toolkit_lite-1.7.7-cp310-cp310-linux_aarch64.whl
7f68939caa3f2caa3c389142a40adc49  rknn_toolkit_lite-1.7.7-cp310-cp310-linux_armv7l.whl
0bcffbcb1133eb56d8690cca3351bc24  rknn_toolkit_lite-1.7.7-cp35-cp35m-linux_aarch64.whl
90d02b34982269c09234bae4cd7f1bba  rknn_toolkit_lite-1.7.7-cp36-cp36m-linux_aarch64.whl
33f33f42ab28bb46926a22173045a8ee  rknn_toolkit_lite-1.7.7-cp36-cp36m-linux_armv7l.whl
a8d00b77e6ab8f3c8301c75628e25e88  rknn_toolkit_lite-1.7.7-cp36-cp36m-linux_x86_64.whl
97a305c21093ec339cfe8318fdbb37a4  rknn_toolkit_lite-1.7.7-cp36-cp36m-macosx_10_15_x86_64.whl
634d4f7add2b1a6b01ae56d8dcd056d4  rknn_toolkit_lite-1.7.7-cp36-cp36m-win_amd64.whl
34b7ea11df05dfc45fc208bf9ab8ebb2  rknn_toolkit_lite-1.7.7-cp37-cp37m-linux_aarch64.whl
29bdb4390a64d0d372f1334819c9313d  rknn_toolkit_lite-1.7.7-cp37-cp37m-linux_armv7l.whl
c52e110a4690c787298853337d1d1fa4  rknn_toolkit_lite-1.7.7-cp37-cp37m-macosx_10_15_x86_64.whl
e181a9d7c9050e151c0f40c04308e228  rknn_toolkit_lite-1.7.7-cp38-cp38-linux_aarch64.whl
3ae2997cac705859cc5bcedc4cae5c62  rknn_toolkit_lite-1.7.7-cp38-cp38-linux_x86_64.whl
029ddbafcd8aa9e328db89058a8963df  rknn_toolkit_lite-1.7.7-cp39-cp39-linux_aarch64.whl
```

# Notes
- Currently rknn-toolkit is not compatible with [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- Latest version: 1.7.7(Release version)

# Feedback and Community Support
- QQ Group Chat: 1025468710 (full, please join group 3)
- QQ Group Chat2: 547021958 (full, please join group 3)
- QQ Group Chat3: 469385426
<center class="half">
  <img width="200" height="200"  src="https://github.com/airockchip/rknn-toolkit/blob/master/QQGroupQRCode.png" title="QQ Group Chat"/>
  <img width="200" height="200"  src="https://github.com/airockchip/rknn-toolkit/blob/master/QQGroup2QRCode.png" title="QQ Group Chat2"/>
  <img width="200" height="200"  src="https://github.com/airockchip/rknn-toolkit/blob/master/QQGroup3QRCode.png" title="QQ Group Chat3"/>
</center>
