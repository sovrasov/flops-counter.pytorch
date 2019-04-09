# Flops counter for convolutional networks in pytorch framework
[![Pypi version](https://img.shields.io/pypi/v/ptflops.svg)](https://pypi.org/project/ptflops/)

This script is designed to compute the theoretical amount of multiply-add operations
in convolutional neural networks. It also can compute the number of parameters and
print per-layer computational cost of a given network.

Supported layers:
- Convolution2d (including grouping)
- BatchNorm2d
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU)
- Linear
- Upsample
- Poolings (AvgPool2d, MaxPool2d and adaptive ones)

Requirements: Pytorch 0.4.1 or 1.0, torchvision 0.2.1

Thanks to @warmspringwinds for the initial version of script.

## Install the latest version
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
  flops, params = get_model_complexity_info(net, (224, 224), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)
```
