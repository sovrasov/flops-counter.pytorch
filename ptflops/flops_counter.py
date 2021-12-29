'''
Copyright (C) 2019-2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial

import torch
import torch.nn as nn

from .pytorch_ops import MODULES_MAPPING, CUSTOM_MODULES_MAPPING


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params, units='GMac',
                           precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1

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
        return ', '.join([params_to_string(accumulated_params_num,
                                           units='M', precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=units, precision=precision),
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
