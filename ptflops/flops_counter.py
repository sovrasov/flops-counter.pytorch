'''
Copyright (C) 2019-2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .pytorch_engine import get_flops_pytorch
from .utils import flops_to_string, params_to_string


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch'):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)

    if backend == 'pytorch':
        flops_count, params_count = get_flops_pytorch(model, input_res,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count
