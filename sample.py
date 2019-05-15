import argparse
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

pt_models = {'resnet18': models.resnet18, 'resnet50': models.resnet50,
             'alexnet': models.alexnet,
             'vgg16': models.vgg16,
             'squeezenet': models.squeezenet1_0,
             'densenet': models.densenet161,
             'inception': models.inception_v3}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flops counter sample script.')
    parser.add_argument('--device', type=int, default=-1, help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()), type=str, default='resnet18')
    args = parser.parse_args()

    with torch.cuda.device(args.device):
        net = pt_models[args.model]()
        flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
        print('Flops: ' + flops)
        print('Params: ' + params)
