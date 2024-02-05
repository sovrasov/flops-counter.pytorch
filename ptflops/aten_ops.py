'''
Copyright (C) 2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

from typing import Any, List

import torch

aten = torch.ops.aten


def get_shape(i):
    return i.shape


def prod(x) -> int:
    res = 1
    for i in x:
        res *= i
    return res


def matmul_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def addmm_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def bmm_flop(inputs: List[Any], outputs: List[Any]) -> int:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensors.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
    bias: bool = False,
) -> int:
    """
    Count MACs for convolution.
    Summation is ignored when applying conv kernel, but counted for bias.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
        bias (bool): is the bias counted
    Returns:
        int: the number of MACs
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    if bias:
        flop += batch_size * out_shape[1] * prod(out_shape[2:])
    return flop


def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    (input, w, b, stride, pad, dilation,
     transposed, _, groups) = inputs
    output = outputs[0]
    return conv_flop_count(input.shape, w.shape, output.shape,
                           transposed=transposed, bias=b is not None)


ATEN_OPS_MAPPING = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
}
