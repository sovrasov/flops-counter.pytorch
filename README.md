# Flops counter for convolutional networks in pytorch framework
[![Pypi version](https://img.shields.io/pypi/v/ptflops.svg)](https://pypi.org/project/ptflops/)

This script is designed to compute the theoretical amount of multiply-add operations
in convolutional neural networks. It also can compute the number of parameters and
print per-layer computational cost of a given network.

Supported layers:
- Conv1d/2d/3d (including grouping)
- ConvTranspose2d (including grouping)
- BatchNorm1d/2d/3d
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU)
- Linear
- Upsample
- Poolings (AvgPool1d/2d/3d, MaxPool1d/2d/3d and adaptive ones)

Requirements: Pytorch >= 0.4.1, torchvision >= 0.2.1

Thanks to @warmspringwinds for the initial version of script.

## Usage tips

- This script doesn't take into account `torch.nn.functional.*` operations. For an instance, if one have a semantic segmentation model and use `torch.nn.functional.interpolate` to upscale features, these operations won't contribute to overall amount of flops. To avoid that one can use `torch.nn.Upsample` instead of `torch.nn.functional.interpolate`.
- `ptflops` launches a given model on a random tensor and estimates amount of computations during inference. Complicated models can have several inputs, some of them could be optional. To construct non-trivial input one can use the `input_constructor` argument of the `get_model_complexity_info`. `input_constructor` is a function that takes the input spatial resolution as a tuple and returns a dict with named input arguments of the model. Next this dict would be passed to the model as keyworded arguments.

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
  flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)
```

## Benchmark

### [torchvision](https://pytorch.org/docs/1.0.0/torchvision/models.html)

Model         | Input Resolution | Params(M) | MACs(G) | Top-1 error | Top-5 error
---           |---               |---        |---      |---          |---
alexnet       |224x224           | 61.1      | 0.72    | 43.45       | 20.91
vgg11         |224x224           | 132.86    | 7.63    | 30.98       | 11.37
vgg13         |224x224           | 133.05    | 11.34   | 30.07       | 10.75
vgg16         |224x224           | 138.36    | 15.5    | 28.41       | 9.62
vgg19         |224x224           | 143.67    | 19.67   | 27.62       | 9.12
vgg11_bn      |224x224           | 132.87    | 7.64    | 29.62       | 10.19
vgg13_bn      |224x224           | 133.05    | 11.36   | 28.45       | 9.63
vgg16_bn      |224x224           | 138.37    | 15.53   | 26.63       | 8.50
vgg19_bn      |224x224           | 143.68    | 19.7    | 25.76       | 8.15
resnet18      |224x224           | 11.69     | 1.82    | 30.24       | 10.92
resnet34      |224x224           | 21.8      | 3.68    | 26.70       | 8.58
resnet50      |224x224           | 25.56     | 4.12    | 23.85       | 7.13
resnet101     |224x224           | 44.55     | 7.85    | 22.63       | 6.44
resnet152     |224x224           | 60.19     | 11.58   | 21.69       | 5.94
squeezenet1_0 |224x224           | 1.25      | 0.83    | 41.90       | 19.58
squeezenet1_1 |224x224           | 1.24      | 0.36    | 41.81       | 19.38
densenet121   |224x224           | 7.98      | 2.88    | 25.35       | 7.83
densenet169   |224x224           | 14.15     | 3.42    | 24.00       | 7.00
densenet201   |224x224           | 20.01     | 4.37    | 22.80       | 6.43
densenet161   |224x224           | 28.68     | 7.82    | 22.35       | 6.20
inception_v3  |224x224           | 27.16     | 2.85    | 22.55       | 6.44

* Top-1 error - ImageNet single-crop top-1 error (224x224)
* Top-5 error - ImageNet single-crop top-5 error (224x224)

### [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

Model               | Input Resolution | Params(M) | MACs(G)     | Acc@1       | Acc@5
---                 |---               |---        |---          |---          |---
alexnet             | 224x224          | 61.1      | 0.72        | 56.432      | 79.194
bninception         | 224x224          | 11.3      | 2.05        | 73.524      | 91.562
cafferesnet101      | 224x224          | 44.55     | 7.62        | 76.2        | 92.766
densenet121         | 224x224          | 7.98      | 2.88        | 74.646      | 92.136
densenet161         | 224x224          | 28.68     | 7.82        | 77.56       | 93.798
densenet169         | 224x224          | 14.15     | 3.42        | 76.026      | 92.992
densenet201         | 224x224          | 20.01     | 4.37        | 77.152      | 93.548
dpn107              | 224x224          | 86.92     | 18.42       | 79.746      | 94.684
dpn131              | 224x224          | 79.25     | 16.13       | 79.432      | 94.574
dpn68               | 224x224          | 12.61     | 2.36        | 75.868      | 92.774
dpn68b              | 224x224          | 12.61     | 2.36        | 77.034      | 93.59
dpn92               | 224x224          | 37.67     | 6.56        | 79.4        | 94.62
dpn98               | 224x224          | 61.57     | 11.76       | 79.224      | 94.488
fbresnet152         | 224x224          | 60.27     | 11.6        | 77.386      | 93.594
inceptionresnetv2   | 299x299          | 55.84     | 13.22       | 80.17       | 95.234
inceptionv3         | 299x299          | 27.16     | 5.73        | 77.294      | 93.454
inceptionv4         | 299x299          | 42.68     | 12.31       | 80.062      | 94.926
nasnetalarge        | 331x331          | 88.75     | 24.04       | 82.566      | 96.086
nasnetamobile       | 224x224          | 5.29      | 0.59        | 74.08       | 91.74
pnasnet5large       | 331x331          | 86.06     | 25.21       | 82.736      | 95.992
polynet             | 331x331          | 95.37     | 34.9        | 81.002      | 95.624
resnet101           | 224x224          | 44.55     | 7.85        | 77.438      | 93.672
resnet152           | 224x224          | 60.19     | 11.58       | 78.428      | 94.11
resnet18            | 224x224          | 11.69     | 1.82        | 70.142      | 89.274
resnet34            | 224x224          | 21.8      | 3.68        | 73.554      | 91.456
resnet50            | 224x224          | 25.56     | 4.12        | 76.002      | 92.98
resnext101_32x4d    | 224x224          | 44.18     | 8.03        | 78.188      | 93.886
resnext101_64x4d    | 224x224          | 83.46     | 15.55       | 78.956      | 94.252
se_resnet101        | 224x224          | 49.33     | 7.63        | 78.396      | 94.258
se_resnet152        | 224x224          | 66.82     | 11.37       | 78.658      | 94.374
se_resnet50         | 224x224          | 28.09     | 3.9         | 77.636      | 93.752
se_resnext101_32x4d | 224x224          | 48.96     | 8.05        | 80.236      | 95.028
se_resnext50_32x4d  | 224x224          | 27.56     | 4.28        | 79.076      | 94.434
senet154            | 224x224          | 115.09    | 20.82       | 81.304      | 95.498
squeezenet1_0       | 224x224          | 1.25      | 0.83        | 58.108      | 80.428
squeezenet1_1       | 224x224          | 1.24      | 0.36        | 58.25       | 80.8
vgg11               | 224x224          | 132.86    | 7.63        | 68.97       | 88.746
vgg11_bn            | 224x224          | 132.87    | 7.64        | 70.452      | 89.818
vgg13               | 224x224          | 133.05    | 11.34       | 69.662      | 89.264
vgg13_bn            | 224x224          | 133.05    | 11.36       | 71.508      | 90.494
vgg16               | 224x224          | 138.36    | 15.5        | 71.636      | 90.354
vgg16_bn            | 224x224          | 138.37    | 15.53       | 73.518      | 91.608
vgg19               | 224x224          | 143.67    | 19.67       | 72.08       | 90.822
vgg19_bn            | 224x224          | 143.68    | 19.7        | 74.266      | 92.066
xception            | 299x299          | 22.86     | 8.42        | 78.888      | 94.292

* Acc@1 - ImageNet single-crop top-1 accuracy on validation images of the same size used during the training process.
* Acc@5 - ImageNet single-crop top-5 accuracy on validation images of the same size used during the training process.
