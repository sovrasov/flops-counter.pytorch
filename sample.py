import argparse
import torchvision.models as models
import torch.nn as nn
import torch
from flops_counter import add_flops_counting_methods

def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + 'GMac'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + 'MMac'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + 'KMac'
    return str(flops) + 'Mac'

def get_model_parameters_number(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + 'M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + 'k'

    return str(params_num)

pt_models = { 'resnet18': models.resnet18, 'resnet50': models.resnet50,
              'alexnet': models.alexnet,
              'vgg16': models.vgg16, 'squeezenet': models.squeezenet1_0,
              'densenet': models.densenet161}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for Face Recognition in PyTorch')
    parser.add_argument('--device', type=int, default=-1, help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()), type=str, default='resnet18')
    args = parser.parse_args()

    with torch.cuda.device(args.device):
        net = pt_models[args.model]()
        batch = torch.FloatTensor(1, 3, 224, 224)
        model = add_flops_counting_methods(net)
        model.eval().start_flops_count()
        _ = model(batch)

        #print(model)
        print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
        print('Params: ' + get_model_parameters_number(model))
