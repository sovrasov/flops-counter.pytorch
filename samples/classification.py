import argparse
import sys

import torch
import torchvision
from packaging import version
from torchvision import models as models

from ptflops import get_model_complexity_info

pt_models = {'resnet18': models.resnet18,
             'resnet50': models.resnet50,
             'alexnet': models.alexnet,
             'vgg16': models.vgg16,
             'squeezenet': models.squeezenet1_0,
             'densenet': models.densenet161,
             'inception': models.inception_v3,
             'convnext_base': models.convnext_base}

if version.parse(torchvision.__version__) > version.parse('0.15'):
    pt_models['vit_b_16'] = models.vit_b_16


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='resnet18')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    net = pt_models[args.model]()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    macs, params = get_model_complexity_info(net, (3, 224, 224),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             ost=ost)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
