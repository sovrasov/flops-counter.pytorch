import argparse
import torchvision.models as models
import torch.nn as nn
import torch
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

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
        out = model(batch)

        print(model)
        print('Output shape: {}'.format(list(out.shape)))
        print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
        print('Params: ' + get_model_parameters_number(model))
