# Flops counter for convolutional networks in pytorch framework

Requirements: Pytorch 0.4.1, torchvision 0.2.1

Thanks to @warmspringwinds for the initial version of script.

Supported layers:
- Convolution2d (including grouping)
- BatchNorm2d
- Activations (ReLU, PReLU, ELU, ReLU6, LeakyReLU)
- Linear
- Upsample
- Poolings (AvgPool2d and MaxPool2d)

## Example
```python
import torchvision.models as models
import torch.nn as nn
import torch
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

with torch.cuda.device(0):
  net = models.densenet161()
  batch = torch.FloatTensor(1, 3, 224, 224)
  model = add_flops_counting_methods(net)
  model.eval().start_flops_count()
  out = model(batch)

  print(model)
  print('Output shape: {}'.format(list(out.shape)))
  print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
  print('Params: ' + get_model_parameters_number(model))
```
