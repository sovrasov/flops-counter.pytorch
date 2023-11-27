'''
Copyright (C) 2021-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    input_last_dim = input.shape[-1]
    pre_last_dims_prod = np.prod(input.shape[0:-1], dtype=np.int64)
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int((input_last_dim * output_last_dim + bias_flops)
                            * pre_last_dims_prod)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape, dtype=np.int64))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape, dtype=np.int64)
    if hasattr(module, "affine") and module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output, extra_per_position_flops=0):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims, dtype=np.int64)) * \
        (in_channels * filters_per_channel + extra_per_position_flops)

    active_elements_count = batch_size * int(np.prod(output_dims, dtype=np.int64))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def deformable_conv_flops_counter_hook(conv_module, input, output):
    # 20 = 4 x 5 is an approximate cost of billinear interpolation, 2x2 grid is used
    # 4 is an approximate cost of fractional coordinates computation
    deformable_conv_extra_complexity = 20 + 4
    # consider also modulation multiplication
    if len(input) == 3 and input[2] is not None:
        deformable_conv_extra_complexity += 1
    conv_flops_counter_hook(conv_module, input, output, deformable_conv_extra_complexity)


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


def timm_attention_counter_hook(attention_module, input, output):
    flops = 0
    B, N, C = input[0].shape  # [Batch_size, Seq_len, Dimension]

    # QKV projection is already covered in MODULES_MAPPING

    # Q scaling
    flops += N * attention_module.head_dim * attention_module.num_heads

    # head flops
    head_flops = (
        (N * N * attention_module.head_dim)  # QK^T
        + (N * N)  # softmax
        + (N * N * attention_module.head_dim)  # AV
    )
    flops += head_flops * attention_module.num_heads

    # Final projection is already covered in MODULES_MAPPING

    flops *= B
    attention_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,

    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    nn.LayerNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook

try:
    import torchvision.ops as tops
    MODULES_MAPPING[tops.DeformConv2d] = deformable_conv_flops_counter_hook
except ImportError:
    pass

try:
    from timm.models.vision_transformer import Attention as timm_Attention
    MODULES_MAPPING[timm_Attention] = timm_attention_counter_hook
except ImportError:
    pass


def _linear_functional_flops_hook(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    if bias is not None:
        macs += out_features
    return macs


def _numel_functional_flops_hook(input, *args, **kwargs):
    return input.numel()


def _interpolate_functional_flops_hook(*args, **kwargs):
    input = kwargs.get('input', None)
    if input is None and len(args) > 0:
        input = args[0]

    size = kwargs.get('size', None)
    if size is None and len(args) > 1:
        size = args[1]

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(np.prod(size, dtype=np.int64))
        else:
            return int(size)

    scale_factor = kwargs.get('scale_factor', None)
    if scale_factor is None and len(args) > 2:
        scale_factor = args[2]
    assert scale_factor is not None, "either size or scale_factor"
    "should be passes to interpolate"

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops *= int(np.prod(scale_factor, dtype=np.int64))
    else:
        flops *= scale_factor**len(input)

    return flops


def _matmul_tensor_flops_hook(input, other, *args, **kwargs):
    flops = np.prod(input.shape, dtype=np.int64) * other.shape[-1]
    return flops


def _addmm_tensor_flops_hook(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    flops = np.prod(mat1.shape, dtype=np.int64) * mat2.shape[-1]
    if beta != 0:
        flops += np.prod(input.shape, dtype=np.int64)
    return flops


def _elementwise_tensor_flops_hook(input, other, *args, **kwargs):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return np.prod(other.shape, dtype=np.int64)
        else:
            return 1
    elif not torch.is_tensor(other):
        return np.prod(input.shape, dtype=np.int64)
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = np.prod(final_shape, dtype=np.int64)
        return flops


FUNCTIONAL_MAPPING = {
    F.linear: _linear_functional_flops_hook,
    F.relu: _numel_functional_flops_hook,
    F.prelu: _numel_functional_flops_hook,
    F.elu: _numel_functional_flops_hook,
    F.relu6: _numel_functional_flops_hook,
    F.gelu: _numel_functional_flops_hook,

    F.avg_pool1d: _numel_functional_flops_hook,
    F.avg_pool2d: _numel_functional_flops_hook,
    F.avg_pool3d: _numel_functional_flops_hook,
    F.max_pool1d: _numel_functional_flops_hook,
    F.max_pool2d: _numel_functional_flops_hook,
    F.max_pool3d: _numel_functional_flops_hook,
    F.adaptive_avg_pool1d: _numel_functional_flops_hook,
    F.adaptive_avg_pool2d: _numel_functional_flops_hook,
    F.adaptive_avg_pool3d: _numel_functional_flops_hook,
    F.adaptive_max_pool1d: _numel_functional_flops_hook,
    F.adaptive_max_pool2d: _numel_functional_flops_hook,
    F.adaptive_max_pool3d: _numel_functional_flops_hook,

    F.softmax: _numel_functional_flops_hook,

    F.upsample: _interpolate_functional_flops_hook,
    F.interpolate: _interpolate_functional_flops_hook,
}

if hasattr(F, "silu"):
    FUNCTIONAL_MAPPING[F.silu] = _numel_functional_flops_hook


TENSOR_OPS_MAPPING = {
    torch.matmul: _matmul_tensor_flops_hook,
    torch.Tensor.matmul: _matmul_tensor_flops_hook,
    torch.mm: _matmul_tensor_flops_hook,
    torch.Tensor.mm: _matmul_tensor_flops_hook,
    torch.bmm: _matmul_tensor_flops_hook,
    torch.Tensor.bmm: _matmul_tensor_flops_hook,

    torch.addmm: _addmm_tensor_flops_hook,
    torch.baddbmm: _addmm_tensor_flops_hook,
    torch.Tensor.addmm: _addmm_tensor_flops_hook,

    torch.mul: _elementwise_tensor_flops_hook,
    torch.Tensor.mul: _elementwise_tensor_flops_hook,
    torch.add: _elementwise_tensor_flops_hook,
    torch.Tensor.add: _elementwise_tensor_flops_hook,
}
