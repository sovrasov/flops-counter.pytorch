'''
Copyright (C) 2019-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import torch.nn as nn

from .pytorch_engine import get_flops_pytorch
from .utils import flops_to_string, params_to_string


def get_model_complexity_info(model: nn.Module,
                              input_res: Tuple[int, ...],
                              print_per_layer_stat: bool = True,
                              as_strings: bool = True,
                              input_constructor: Optional[Callable[[Tuple], Dict]] = None,
                              ost: TextIO = sys.stdout,
                              verbose: bool = False,
                              ignore_modules: List[nn.Module] = [],
                              custom_modules_hooks: Dict[nn.Module, Any] = {},
                              backend: str = 'pytorch',
                              flops_units: Optional[str] = None,
                              param_units: Optional[str] = None,
                              output_precision: int = 2) -> Tuple[Union[str, int, None],
                                                                  Union[str, int, None]]:
    """
    Analyzes the input model and collects the amounts of parameters and MACs
    required to make a forward pass of the model.

    :param model: Input model to analyze
    :type model: nn.Module
    :param input_res: A tuple that sets the input resolution for the model. Batch
        dimension is added automatically: (3, 224, 224) -> (1, 3, 224, 224).
    :type input_res: Tuple[int, ...]
    :param print_per_layer_stat: Flag to enable or disable printing of per-layer
        MACs/params statistics. This feature works only for layers derived
        from torch.nn.Module. Other operations are ignored.
    :type print_per_layer_stat: bool
    :param as_strings: Flag that allows to get ready-to-print string representation
        of the final params/MACs estimations. Otherwise, a tuple with raw numbers
        will be returned.
    :type as_strings: bool
    :param input_constructor: A callable that takes the :input_res parameter and
        returns an output suitable for the model. It can be used if model requires
        more than one input tensor or any other kind of irregular input.
    :type input_constructor: Optional[Callable[[Tuple], Dict]]
    :param ost: A stream to print output.
    :type ost: TextIO
    :param verbose: Parameter to control printing of extra information and warnings.
    :type verbose: bool
    :param ignore_modules: A list of torch.nn.Module modules to ignore.
    :type ignore_modules: nn.Module
    :param custom_modules_hooks: A dict that contains custom hooks on torch modules.
    :type custom_modules_hooks: Dict[nn.Module, Any]
    :param flops_units: Units for string representation of MACs (GMac, MMac or KMac).
    :type flops_units: Optional[str]
    :param param_units: Units for string representation of params (M, K or B).
    :type param_units: Optional[str]
    :param output_precision: Floating point precision for representing MACs/params in
        given units.
    :type output_precision: int

    Returns:
        Tuple[Union[str, int, None], Union[str, int, None]]: Return value is a tuple
            (macs, params): Nones in case of a failure during computations, or
            strings if :as_strings is true or integers otherwise.
    """
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
