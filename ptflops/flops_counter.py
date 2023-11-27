'''
Copyright (C) 2019-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from typing import Any, Callable, Dict, TextIO, Tuple, Union

import torch.nn as nn

from .pytorch_engine import get_flops_pytorch
from .utils import flops_to_string, params_to_string


def get_model_complexity_info(model: nn.Module, input_res: Tuple[int, ...],
                              print_per_layer_stat: bool = True,
                              as_strings: bool = True,
                              input_constructor: Union[Callable, None] = None,
                              ost: TextIO = sys.stdout,
                              verbose: bool = False, ignore_modules=[],
                              custom_modules_hooks: Dict[nn.Module, Any] = {},
                              backend: str = 'pytorch',
                              flops_units: Union[str, None] = None,
                              param_units: Union[str, None] = None,
                              output_precision: int = 2):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)

    if backend == 'pytorch':
        flops_count, params_count = get_flops_pytorch(model, input_res,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      flops_units=flops_units,
                                                      param_units=param_units)
    else:
        raise ValueError('Wrong backend name')

    if as_strings and flops_count is not None and params_count is not None:
        flops_string = flops_to_string(
            flops_count,
            units=flops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return flops_string, params_string

    return flops_count, params_count
