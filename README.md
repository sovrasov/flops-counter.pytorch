# Flops counting tool for neural networks in pytorch framework
[![Pypi version](https://img.shields.io/pypi/v/ptflops.svg)](https://pypi.org/project/ptflops/)
[![Build Status](https://travis-ci.com/sovrasov/flops-counter.pytorch.svg?branch=master)](https://travis-ci.com/sovrasov/flops-counter.pytorch)

This script is designed to compute the theoretical amount of multiply-add operations
in convolutional neural networks. It can also compute the number of parameters and
print per-layer computational cost of a given network.

Supported layers:
- Conv1d/2d/3d (including grouping)
- ConvTranspose1d/2d/3d (including grouping)
- BatchNorm1d/2d/3d, GroupNorm, InstanceNorm1d/2d/3d, LayerNorm
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU, GELU)
- Linear
- Upsample
- Poolings (AvgPool1d/2d/3d, MaxPool1d/2d/3d and adaptive ones)

Experimental support:
- RNN, LSTM, GRU (NLH layout is assumed)
- RNNCell, LSTMCell, GRUCell
- torch.nn.MultiheadAttention
- torchvision.ops.DeformConv2d
- visual transformers from [timm](https://github.com/huggingface/pytorch-image-models)

Requirements: Pytorch >= 1.1, torchvision >= 0.3

Thanks to @warmspringwinds for the initial version of script.

## Usage tips

- This tool doesn't take into account some of the `torch.nn.functional.*` and `tensor.*` operations. Therefore unsupported operations are
not contributing to the final complexity estimation. See `ptflops/pytorch_ops.py:FUNCTIONAL_MAPPING,TENSOR_OPS_MAPPING` to check supported ops.
- `ptflops` launches a given model on a random tensor and estimates amount of computations during inference. Complicated models can have several inputs, some of them could be optional. To construct non-trivial input one can use the `input_constructor` argument of the `get_model_complexity_info`. `input_constructor` is a function that takes the input spatial resolution as a tuple and returns a dict with named input arguments of the model. Next this dict would be passed to the model as a keyword arguments.
- `verbose` parameter allows to get information about modules that don't contribute to the final numbers.
- `ignore_modules` option forces `ptflops` to ignore the listed modules. This can be useful
for research purposes. For instance, one can drop all convolutions from the counting process
specifying `ignore_modules=[torch.nn.Conv2d]`.

## Install the latest version
From PyPI:
```bash
pip install ptflops
```

From this repository:
```bash
pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
```

## Example
```python
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.densenet161()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
```

## Citation
If ptflops was useful for your paper or tech report, please cite me:
```
@online{ptflops,
  author = {Vladislav Sovrasov},
  title = {ptflops: a flops counting tool for neural networks in pytorch framework},
  year = 2018-2023,
  url = {https://github.com/sovrasov/flops-counter.pytorch},
}
```

## Benchmark

### [torchvision](https://pytorch.org/vision/0.16/models.html)

Model                  | Input Resolution | Params(M) | MACs(G)
---                    |---               |---        |---
alexnet                | 224x224          | 61.10     | 0.72
convnext_base          | 224x224          | 88.59     | 15.43
densenet121            | 224x224          | 7.98      | 2.90
efficientnet_b0        | 224x224          | 5.29      | 0.41
efficientnet_v2_m      | 224x224          | 54.14     | 5.43
googlenet              | 224x224          | 13.00     | 1.51
inception_v3           | 224x224          | 27.16     | 2.86
maxvit_t               | 224x224          | 30.92     | 5.48
mnasnet1_0             | 224x224          | 4.38      | 0.33
mobilenet_v2           | 224x224          | 3.50      | 0.32
mobilenet_v3_large     | 224x224          | 5.48      | 0.23
regnet_y_1_6gf         | 224x224          | 11.20     | 1.65
resnet18               | 224x224          | 11.69     | 1.83
resnet50               | 224x224          | 25.56     | 4.13
resnext50_32x4d        | 224x224          | 25.03     | 4.29
shufflenet_v2_x1_0     | 224x224          | 2.28      | 0.15
squeezenet1_0          | 224x224          | 1.25      | 0.84
vgg16                  | 224x224          | 138.36    | 15.52
vit_b_16               | 224x224          | 86.57     | 17.60
wide_resnet50_2        | 224x224          | 68.88     | 11.45
