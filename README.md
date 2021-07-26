(第一次写markdown,不熟悉,见谅

准备一下:
拉取docker:(需要魔法)
>sudo docker pull zepan/zhouyi

**1.模型：**
在[onnx/models/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)仓库里下载的ResNet50-caffe2

**2.矫正数据集**
注意，因为onnx提供的模型是做了归一化处理(昨天被坑了好久,看到群里一个叫'IF'的人的提示才翻然醒悟)，所以这里也要，我使用了torchvision的
>transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

官方提供了脚本文件，但是我不喜欢用cv2,所以随手写了一个用torchvision的脚本
```
import os
from torchvision import transforms
from PIL import Image
import numpy as np

imgs_path = './img/'
imgs_list = os.listdir(imgs_path)
imgs_path_list = [imgs_path + i for i in imgs_list]
imgs_list = []
for i in imgs_path_list:
    imgs_list = imgs_list + [Image.open(i)]
transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
imgs_list = [np.array(transforms(i)) for i in imgs_list]
imgs_list = [np.transpose(i,(1,2,0)) for i in imgs_list]
imgs_list = np.array(imgs_list)
print(imgs_list.shape)
np.save('./out/data.npy',imgs_list)

#保存label
label_array = []
with open('val.txt') as f:
    line = f.readlines()
    for i in range(imgs_list.shape[0]):
        label_array = label_array + [line[i][29:-2]]
label_array = [int(i) for i in label_array]
label_array = np.array(label_array)
print(label_array.shape)
np.save('./out/label.npy',label_array)
```
比官方的简洁多了(bushi)
我使用了imagenet2012的前99张图片作为矫正集(为什么是99?因为有一张黑白的懒得我处理，索性删掉了)
我知道只要5张就行，但是我下载数据集这么久，当然要多级张才爽

**3.输入数据**
官方提供的input也要经过处理，这里用了case小姐姐的脚本:
```
import cv2
import numpy as np

input_height=224
input_width=224
input_channel=3

img_path = "../preprocess_shufflenet_dataset/img/ILSVRC2012_val_00000004.JPEG"

orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (input_width, input_height))
image = (image - 127.5) / 1
image = np.expand_dims(image, axis=0)
image = image.astype(np.int8)

image.tofile("input.bin")
print("save to input.bin OK")
```
**4.编辑run.cfg:**
```
[Common]
mode = run

[Parser]
model_type = onnx
input_data_format = NCHW
model_name = shufflenet
detection_postprocess = 
model_domain = image_classification
input_model = ./input/resnet.onnx
input = gpu_0/data_0
input_shape = [1, 3, 224, 224]
output = gpu_0/softmax_1

[AutoQuantizationTool]
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
reverse_rgb = False
calibration_data = ./input/dataset.npy
calibration_label = ./input/label.npy
label_id_offset = 0
preprocess_mode = normalize
quant_precision = int8

[GBuilder]
inputs=./input/input.bin
simulator=aipu_simulator_z1
outputs=output.bin
profile= True
target=Z1_0701
```
**5.仿真**
使用aipubuild进行仿真
>aipubuild ./input/run.cfg

~~过程太长了，直接放quant_predict.py解析的结果吧~~
好像一定要输出结果
```
root@14912260e482:~/resnet_onnx# aipubuild input/run.cfg
WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

[I] Parsing model....
[I] [Parser]: Begin to parse onnx model shufflenet...
2021-07-18 05:00:16.769472: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2021-07-18 05:00:16.832554: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
2021-07-18 05:00:16.833385: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x11286360 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-07-18 05:00:16.833450: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
[I] [Parser]: Parser done!
[I] Parse model complete
[I] Quantizing model....
[I] AQT start: model_name:shufflenet, calibration_method:MEAN, batch_size:1
[I] ==== read ir ================
[I] 	float32 ir txt: /tmp/AIPUBuilder_1626584414.050215/shufflenet.txt
[I] 	float32 ir bin2: /tmp/AIPUBuilder_1626584414.050215/shufflenet.bin
[I] ==== read ir DONE.===========
WARNING:tensorflow:From /usr/local/bin/aipubuild:8: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /usr/local/bin/aipubuild:8: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From /usr/local/bin/aipubuild:8: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.

[I] ==== auto-quantization ======
WARNING:tensorflow:From /usr/local/bin/aipubuild:8: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
Instructions for updating:
Use eager execution and: 
`tf.data.TFRecordDataset(path)`
WARNING:tensorflow:Entity <bound method ImageNet.data_transform_fn of <AIPUBuilder.AutoQuantizationTool.auto_quantization.data_set.ImageNet object at 0x7f5dec875e48>> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: <cyfunction ImageNet.data_transform_fn at 0x7f5e20ce5d38> is not a module, class, method, function, traceback, frame, or code object
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/impl/api.py:330: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/impl/api.py:330: The name tf.parse_single_example is deprecated. Please use tf.io.parse_single_example instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/func_graph.py:915: The name tf.image.resize_images is deprecated. Please use tf.image.resize instead.

WARNING:tensorflow:From /usr/local/bin/aipubuild:8: DatasetV1.make_one_shot_iterator (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `for ... in dataset:` to iterate over a dataset. If using `tf.estimator`, return the `Dataset` object directly from your input function. As a last resort, you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)`.
WARNING:tensorflow:From /usr/local/bin/aipubuild:8: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From /usr/local/bin/aipubuild:8: The name tf.local_variables_initializer is deprecated. Please use tf.compat.v1.local_variables_initializer instead.


[I] 	step1: get max/min statistic value DONE
[I] 	step2: quantization each op DONE
[I] 	step3: build quantization forward DONE
[I] 	step4: show output scale of end node:
[I] 		layer_id: 75, layer_top:gpu_0/softmax_1, output_scale:[406.21088]
[I] ==== auto-quantization DONE =
[I] Quantize model complete
[I] Building ...
[I] [common_options.h: 276] BuildTool version: 4.0.175. Build for target Z1_0701 at frequency 800MHz
[I] [common_options.h: 297] using default profile events to profile AIFF

[I] [IRChecker] Start to check IR: /tmp/AIPUBuilder_1626584414.050215/shufflenet_int8.txt
[I] [IRChecker] model_name: shufflenet
[I] [IRChecker] IRChecker: All IR pass
[I] [graph.cpp : 846] loading graph weight: /tmp/AIPUBuilder_1626584414.050215/shufflenet_int8.bin size: 0x186d9a4
[I] [builder.cpp:1059] Total memory for this graph: 0x1f10000 Bytes
[I] [builder.cpp:1060] Text   section:	0x00030300 Bytes
[I] [builder.cpp:1061] RO     section:	0x00003b00 Bytes
[I] [builder.cpp:1062] Desc   section:	0x00006c00 Bytes
[I] [builder.cpp:1063] Data   section:	0x0186de00 Bytes
[I] [builder.cpp:1064] BSS    section:	0x00627400 Bytes
[I] [builder.cpp:1065] Stack         :	0x00040400 Bytes
[I] [builder.cpp:1066] Workspace(BSS):	0x000c4000 Bytes
[I] [main.cpp  : 467] # autogenrated by aipurun, do NOT modify!
LOG_FILE=log_default
FAST_FWD_INST=0
INPUT_INST_CNT=1
INPUT_DATA_CNT=2
CONFIG=Z1-0701
LOG_LEVEL=0
INPUT_INST_FILE0=/tmp/temp_1ad93be83224c4f55c75eb41324ca.text
INPUT_INST_BASE0=0x0
INPUT_INST_STARTPC0=0x0
INPUT_DATA_FILE0=/tmp/temp_1ad93be83224c4f55c75eb41324ca.ro
INPUT_DATA_BASE0=0x10000000
INPUT_DATA_FILE1=/tmp/temp_1ad93be83224c4f55c75eb41324ca.data
INPUT_DATA_BASE1=0x20000000
OUTPUT_DATA_CNT=2
OUTPUT_DATA_FILE0=output.bin
OUTPUT_DATA_BASE0=0x22000000
OUTPUT_DATA_SIZE0=0x3e8
OUTPUT_DATA_FILE1=profile_data.bin
OUTPUT_DATA_BASE1=0x21d00000
OUTPUT_DATA_SIZE1=0x800
RUN_DESCRIPTOR=BIN[0]

[I] [main.cpp  : 118] run simulator:
aipu_simulator_z1 /tmp/temp_1ad93be83224c4f55c75eb41324ca.cfg
[INFO]:SIMULATOR START!
[INFO]:========================================================================
[INFO]:                             STATIC CHECK
[INFO]:========================================================================
[INFO]:  INST START ADDR : 0x0(0)
[INFO]:  INST END ADDR   : 0x302ff(197375)
[INFO]:  INST SIZE       : 0x30300(197376)
[INFO]:  PACKET CNT      : 0x3030(12336)
[INFO]:  INST CNT        : 0xc0c0(49344)
[INFO]:------------------------------------------------------------------------
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x44e: 0x472021b(POP R27,Rc7) vs 0x5f00000(MVI R0,0x0,Rc7), PACKET:0x44e(1102) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x45b: 0x472021b(POP R27,Rc7) vs 0x5f00000(MVI R0,0x0,Rc7), PACKET:0x45b(1115) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x5c0: 0x472021b(POP R27,Rc7) vs 0x9f80020(ADD.S R0,R0,0x1,Rc7), PACKET:0x5c0(1472) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x7fd: 0x4520180(BRL R0) vs 0x47a03e4(ADD R4,R0,R31,Rc7), PACKET:0x7fd(2045) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x996: 0x4720204(POP R4,Rc7) vs 0x9f80020(ADD.S R0,R0,0x1,Rc7), PACKET:0x996(2454) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0xe40: 0x4720204(POP R4,Rc7) vs 0x47a1be0(ADD R0,R6,R31,Rc7), PACKET:0xe40(3648) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x1266: 0x472021b(POP R27,Rc7) vs 0x5f00000(MVI R0,0x0,Rc7), PACKET:0x1266(4710) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x1273: 0x472021b(POP R27,Rc7) vs 0x5f00000(MVI R0,0x0,Rc7), PACKET:0x1273(4723) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x13d8: 0x472021b(POP R27,Rc7) vs 0x9f80020(ADD.S R0,R0,0x1,Rc7), PACKET:0x13d8(5080) SLOT:0 vs 3
[WARN]:[0803] INST WR/RD REG CONFLICT! PACKET 0x1575: 0x4520180(BRL R0) vs 0x47a03e5(ADD R5,R0,R31,Rc7), PACKET:0x1575(5493) SLOT:0 vs 3
[INFO]:========================================================================
[INFO]:                             STATIC CHECK END
[INFO]:========================================================================

[INFO]:AIPU START RUNNING: BIN[0]
[INFO]:TOTAL TIME: 9.366748s. 
[INFO]:SIMULATOR EXIT!
[I] [main.cpp  : 135] Simulator finished.
Total errors: 0,  warnings: 0
```
**6.解析**
解析结果:
```
predict first 5 label:
    index  231, prob  70, name: Shetland sheepdog, Shetland sheep dog, Shetland
    index  169, prob  25, name: redbone
    index  160, prob   4, name: Rhodesian ridgeback
    index  157, prob   1, name: Blenheim spaniel
    index  224, prob   1, name: schipperke
true first 5 label:
    index  230, prob 109, name: Old English sheepdog, bobtail
    index  231, prob  96, name: Shetland sheepdog, Shetland sheep dog, Shetland
    index  232, prob  57, name: collie
    index  226, prob  54, name: malinois
    index  263, prob  53, name: Brabancon griffon
Detect picture save to result.jpeg
```
**7.提交文件**
>https://drive.google.com/file/d/1l2ayqLYMGjZljdPcRmXFEa0joFSq6vlW/view?usp=sharing
用谷歌盘是因为百度太难用了

第一次发帖欢迎指正
