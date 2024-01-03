'''
Copyright (C) 2024 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''


import sys
import traceback
from collections import defaultdict
from typing import Optional, Tuple, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ptflops.pytorch_engine import get_model_parameters_number
from .aten_ops import ATEN_OPS_MAPPING


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class FlopCounterMode(TorchDispatchMode):
    def __init__(self, module=None):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ['Global']
        self._total_complexity = None
        if module is not None:
            for name, mod in dict(module.named_children()).items():
                mod.register_forward_pre_hook(self.enter_module(name))
                mod.register_forward_hook(self.exit_module(name))

    @property
    def complexity(self):
        return self._total_complexity

    def enter_module(self, name):
        def f(*args):
            self.parents.append(name)
        return f

    def exit_module(self, name):
        def f(*args):
            assert(self.parents[-1] == name)
            self.parents.pop()
        return f

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        self._total_complexity = sum(self.flop_counts['Global'].values())
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in ATEN_OPS_MAPPING:
            flop_count = ATEN_OPS_MAPPING[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count

        return out


def get_flops_aten(model, input_res,
                   print_per_layer_stat=True,
                   input_constructor=None, ost=sys.stdout,
                   verbose=False, ignore_modules=[],
                   custom_modules_hooks={},
                   output_precision=2,
                   flops_units: Optional[str] = 'GMac',
                   param_units: Optional[str] = 'M') -> Tuple[Union[int, None],
                                                              Union[int, None]]:

    params_sum = get_model_parameters_number(model)

    if input_constructor:
        batch = input_constructor(input_res)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(model.parameters()).dtype,
                                             device=next(model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

    try:
        counter = FlopCounterMode(model)
        with counter:
            if isinstance(batch, dict):
                _ = model(**batch)
            else:
                _ = model(batch)
        macs_count = counter.complexity

    except Exception as e:
        print("Flops estimation was not finished successfully because of"
              f"the following exception:\n{type(e)} : {e}")
        traceback.print_exc()

        return None, None

    return macs_count, params_sum
