'''
Copyright (C) 2021-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
import traceback
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pytorch_ops import (CUSTOM_MODULES_MAPPING, FUNCTIONAL_MAPPING,
                          MODULES_MAPPING, TENSOR_OPS_MAPPING)
from .utils import flops_to_string, params_to_string


def get_flops_pytorch(model, input_res,
                      print_per_layer_stat=True,
                      input_constructor=None, ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=2,
                      flops_units: Optional[str] = 'GMac',
                      param_units: Optional[str] = 'M') -> Tuple[Union[int, None],
                                                                 Union[int, None]]:
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)
    if input_constructor:
        batch = input_constructor(input_res)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

    torch_functional_flops = []
    torch_tensor_ops_flops = []
    patch_functional(torch_functional_flops)
    patch_tensor_ops(torch_tensor_ops_flops)

    def reset_environment():
        flops_model.stop_flops_count()
        unpatch_functional()
        unpatch_tensor_ops()
        global CUSTOM_MODULES_MAPPING
        CUSTOM_MODULES_MAPPING = {}

    try:
        if isinstance(batch, dict):
            _ = flops_model(**batch)
        else:
            _ = flops_model(batch)
        flops_count, params_count = flops_model.compute_average_flops_cost()
        flops_count += sum(torch_functional_flops)
        flops_count += sum(torch_tensor_ops_flops)

    except Exception as e:
        print("Flops estimation was not finished successfully because of"
              f"the following exception:\n{type(e)} : {e}")
        traceback.print_exc()
        reset_environment()

        return None, None

    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            flops_count,
            params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    reset_environment()

    return int(flops_count), params_count


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params,
                           flops_units: Optional[str] = 'GMac',
                           param_units: Optional[str] = 'M',
                           precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        if accumulated_params_num > total_params:
            print('Warning: parameters of some of the modules were counted twice because'
                  ' of multiple links to the same modules.'
                  ' Extended per layer parameters num statistic could be unreliable.')

        return ', '.join([params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
            module.__ptflops_backup_flops__ = module.__flops__
            module.__ptflops_backup_params__ = module.__params__
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def remove_flops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__'):
            del module.__flops__
            if hasattr(module, '__ptflops_backup_flops__'):
                module.__flops__ = module.__ptflops_backup_flops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptflops_backup_params__'):
                module.__params__ = module.__ptflops_backup_params__


class torch_function_wrapper:
    def __init__(self, op, handler, collector) -> None:
        self.collector = collector
        self.op = op
        self.handler = handler

    def __call__(self, *args, **kwds):
        flops = self.handler(*args, **kwds)
        self.collector.append(flops)
        return self.op(*args, **kwds)


def patch_functional(collector):
    # F.linear = torch_function_wrapper(F.linear, FUNCTIONAL_MAPPING[F.linear], collector)
    F.relu = torch_function_wrapper(F.relu, FUNCTIONAL_MAPPING[F.relu], collector)
    F.prelu = torch_function_wrapper(F.prelu, FUNCTIONAL_MAPPING[F.prelu], collector)
    F.elu = torch_function_wrapper(F.elu, FUNCTIONAL_MAPPING[F.elu], collector)
    F.relu6 = torch_function_wrapper(F.relu6, FUNCTIONAL_MAPPING[F.relu6], collector)
    F.gelu = torch_function_wrapper(F.gelu, FUNCTIONAL_MAPPING[F.gelu], collector)

    F.avg_pool1d = torch_function_wrapper(F.avg_pool1d,
                                          FUNCTIONAL_MAPPING[F.avg_pool1d], collector)
    F.avg_pool2d = torch_function_wrapper(F.avg_pool2d,
                                          FUNCTIONAL_MAPPING[F.avg_pool2d], collector)
    F.avg_pool3d = torch_function_wrapper(F.avg_pool3d,
                                          FUNCTIONAL_MAPPING[F.avg_pool3d], collector)
    F.max_pool1d = torch_function_wrapper(F.max_pool1d,
                                          FUNCTIONAL_MAPPING[F.max_pool1d], collector)
    F.max_pool2d = torch_function_wrapper(F.max_pool2d,
                                          FUNCTIONAL_MAPPING[F.max_pool2d], collector)
    F.max_pool3d = torch_function_wrapper(F.max_pool3d,
                                          FUNCTIONAL_MAPPING[F.max_pool3d], collector)
    F.adaptive_avg_pool1d = torch_function_wrapper(
        F.adaptive_avg_pool1d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool1d], collector)
    F.adaptive_avg_pool2d = torch_function_wrapper(
        F.adaptive_avg_pool2d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool2d], collector)
    F.adaptive_avg_pool3d = torch_function_wrapper(
        F.adaptive_avg_pool3d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool3d], collector)
    F.adaptive_max_pool1d = torch_function_wrapper(
        F.adaptive_max_pool1d, FUNCTIONAL_MAPPING[F.adaptive_max_pool1d], collector)
    F.adaptive_max_pool2d = torch_function_wrapper(
        F.adaptive_max_pool2d, FUNCTIONAL_MAPPING[F.adaptive_max_pool2d], collector)
    F.adaptive_max_pool3d = torch_function_wrapper(
        F.adaptive_max_pool3d, FUNCTIONAL_MAPPING[F.adaptive_max_pool3d], collector)

    F.softmax = torch_function_wrapper(
        F.softmax, FUNCTIONAL_MAPPING[F.softmax], collector)

    F.upsample = torch_function_wrapper(
        F.upsample, FUNCTIONAL_MAPPING[F.upsample], collector)
    F.interpolate = torch_function_wrapper(
        F.interpolate, FUNCTIONAL_MAPPING[F.interpolate], collector)

    if hasattr(F, "silu"):
        F.silu = torch_function_wrapper(F.silu, FUNCTIONAL_MAPPING[F.silu], collector)


def unpatch_functional():
    # F.linear = F.linear.op
    F.relu = F.relu.op
    F.prelu = F.prelu.op
    F.elu = F.elu.op
    F.relu6 = F.relu6.op
    F.gelu = F.gelu.op
    if hasattr(F, "silu"):
        F.silu = F.silu.op

    F.avg_pool1d = F.avg_pool1d.op
    F.avg_pool2d = F.avg_pool2d.op
    F.avg_pool3d = F.avg_pool3d.op
    F.max_pool1d = F.max_pool1d.op
    F.max_pool2d = F.max_pool2d.op
    F.max_pool3d = F.max_pool3d.op
    F.adaptive_avg_pool1d = F.adaptive_avg_pool1d.op
    F.adaptive_avg_pool2d = F.adaptive_avg_pool2d.op
    F.adaptive_avg_pool3d = F.adaptive_avg_pool3d.op
    F.adaptive_max_pool1d = F.adaptive_max_pool1d.op
    F.adaptive_max_pool2d = F.adaptive_max_pool2d.op
    F.adaptive_max_pool3d = F.adaptive_max_pool3d.op

    F.softmax = F.softmax.op

    F.upsample = F.upsample.op
    F.interpolate = F.interpolate.op


def wrap_tensor_op(op, collector):
    tensor_op_handler = torch_function_wrapper(
        op, TENSOR_OPS_MAPPING[op], collector)

    def wrapper(*args, **kwargs):
        return tensor_op_handler(*args, **kwargs)

    wrapper.op = tensor_op_handler.op

    return wrapper


def patch_tensor_ops(collector):
    torch.matmul = torch_function_wrapper(
        torch.matmul, TENSOR_OPS_MAPPING[torch.matmul], collector)
    torch.Tensor.matmul = wrap_tensor_op(torch.Tensor.matmul, collector)
    torch.mm = torch_function_wrapper(
        torch.mm, TENSOR_OPS_MAPPING[torch.mm], collector)
    torch.Tensor.mm = wrap_tensor_op(torch.Tensor.mm, collector)
    torch.bmm = torch_function_wrapper(
        torch.bmm, TENSOR_OPS_MAPPING[torch.bmm], collector)
    torch.Tensor.bmm = wrap_tensor_op(torch.Tensor.bmm, collector)

    torch.addmm = torch_function_wrapper(
        torch.addmm, TENSOR_OPS_MAPPING[torch.addmm], collector)
    torch.Tensor.addmm = wrap_tensor_op(torch.Tensor.addmm, collector)
    torch.baddbmm = torch_function_wrapper(
        torch.baddbmm, TENSOR_OPS_MAPPING[torch.baddbmm], collector)

    torch.mul = torch_function_wrapper(
        torch.mul, TENSOR_OPS_MAPPING[torch.mul], collector)
    torch.Tensor.mul = wrap_tensor_op(torch.Tensor.mul, collector)
    torch.add = torch_function_wrapper(
        torch.add, TENSOR_OPS_MAPPING[torch.add], collector)
    torch.Tensor.add = wrap_tensor_op(torch.Tensor.add, collector)


def unpatch_tensor_ops():
    torch.matmul = torch.matmul.op
    torch.Tensor.matmul = torch.Tensor.matmul.op
    torch.mm = torch.mm.op
    torch.Tensor.mm = torch.Tensor.mm.op
    torch.bmm = torch.bmm.op
    torch.Tensor.bmm = torch.Tensor.bmm.op

    torch.addmm = torch.addmm.op
    torch.Tensor.addmm = torch.Tensor.addmm.op
    torch.baddbmm = torch.baddbmm.op

    torch.mul = torch.mul.op
    torch.Tensor.mul = torch.Tensor.mul.op
    torch.add = torch.add.op
    torch.Tensor.add = torch.Tensor.add.op
