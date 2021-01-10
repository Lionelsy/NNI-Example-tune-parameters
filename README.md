# NNI-Example (tune parameters)

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

[English](README_EN.md)

[NNI](https://github.com/microsoft/nni)部署下的自动调参模板

## 介绍

>NNI (Neural Network Intelligence) 是一个工具包，可有效的帮助用户设计并调优机器学习模型的神经网络架构，复杂系统的参数（如超参）等。 NNI 的特性包括：易于使用，可扩展，灵活，高效。

本仓库主要介绍了NNI自动调参功能的使用。

相比于原始的训练代码，NNI的自动调参仅需要调整少许代码并添加两个文件即可完成自动调参过程。

笔者视角下主要有以下优点：

- 界面美观，具有单独的网页端展示结果；
- 训练规范，能够简洁地保存训练结果和中间输出；
- 批量训练，对于参数分析实验能够一步完成，十分方便。

## 部署

nni的安装可以参考[NNI官方文档](https://nni.readthedocs.io/zh/latest/Overview.html),在这比主要介绍代码修改部分及两个配置文件[search_space.json](#search_spacejson)和[config.yml](#configyml)。

## 源代码

这里以[example](./example)内PyTorch代码为例(修改自[官方示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-pytorch))。详细对比见[Original.py](./example/Original.py)和[Revised.py](./example/Revised.py)

### 引入依赖

~~~python
import nni
from nni.utils import merge_parameter
~~~

### 修改超参配置

~~~python
tuner_params = nni.get_next_parameter()
params = vars(merge_parameter(get_params(), tuner_params))
~~~

### 训练过程中/结束后返回结果

~~~python
# report intermediate result
nni.report_intermediate_result(test_acc)

# report final result
nni.report_final_result(test_acc)
~~~

按照以上思路修改对应代码部分即可

## search_space.json

官方说明[链接](https://nni.readthedocs.io/zh/latest/Tutorial/SearchSpaceSpec.html)

该文件为希望参数搜索空间，对于单个参数，基本格式如下

~~~json
{
    "参数名称":{"_type":"类型","_value":"取值范围"}
}
~~~



## config.yml

官方说明[链接](https://nni.readthedocs.io/zh/latest/Tutorial/ExperimentConfig.html)

### 指定gpu

~~~config
localConfig:
    gpuIndices: 2,3
~~~



## Tips

### 网页端指定对应端口

~~~bash
nnictl create --config config.yml --port xxxx
~~~



### 报错

#### S1 

Specified GPU index not found

~~~bash
ps aux | grep gpu_metric
显示nni的nvidia-smi的使用情况
kill it
~~~





## License

The entire codebase is under [MIT license](https://github.com/Lionelsy/NNI-PyTorch/blob/main/LICENSE)