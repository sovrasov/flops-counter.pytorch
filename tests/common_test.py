import pytest
import torch
import torch.nn as nn

from ptflops import get_model_complexity_info
from ptflops.flops_counter import FLOPS_BACKEND


class TestOperations:
    @pytest.fixture
    def default_input_image_size(self):
        return (3, 224, 224)

    @pytest.fixture
    def simple_model_mm(self):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.matmul(x.t())

        return CustomModel()

    @pytest.mark.parametrize("backend", [FLOPS_BACKEND.PYTORCH, FLOPS_BACKEND.ATEN])
    def test_conv(self, default_input_image_size, backend: FLOPS_BACKEND):
        net = nn.Sequential(nn.Conv2d(3, 2, 3, bias=True))
        macs, params = get_model_complexity_info(net, default_input_image_size,
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 backend=backend)

        assert params == 3 * 3 * 2 * 3 + 2
        assert macs == 2759904

    @pytest.mark.parametrize("backend", [FLOPS_BACKEND.PYTORCH, FLOPS_BACKEND.ATEN])
    def test_conv_t(self, default_input_image_size, backend: FLOPS_BACKEND):
        net = nn.ConvTranspose2d(3, 2, 3, stride=(2, 2), bias=True)
        macs, params = get_model_complexity_info(net, default_input_image_size,
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 backend=backend)

        assert params == 3 * 3 * 2 * 3 + 2
        assert macs == 3112706

    @pytest.mark.parametrize("backend", [FLOPS_BACKEND.PYTORCH, FLOPS_BACKEND.ATEN])
    def test_fc(self, backend: FLOPS_BACKEND):
        net = nn.Sequential(nn.Linear(3, 2, bias=True))
        macs, params = get_model_complexity_info(net, (3,),
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 backend=backend)

        assert params == 3 * 2 + 2
        assert macs == 8

    @pytest.mark.parametrize("backend", [FLOPS_BACKEND.PYTORCH, FLOPS_BACKEND.ATEN])
    def test_fc_multidim(self, backend: FLOPS_BACKEND):
        net = nn.Sequential(nn.Linear(3, 2, bias=False))
        macs, params = get_model_complexity_info(net, (4, 5, 3),
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 backend=backend)

        assert params == 3 * 2
        assert macs == (3 * 2) * 4 * 5

    def test_input_constructor_tensor(self):
        net = nn.Sequential(nn.Linear(3, 2, bias=True))

        def input_constructor(input_res):
            return torch.ones(()).new_empty((1, *input_res))

        macs, params = get_model_complexity_info(net, (3,),
                                                 input_constructor=input_constructor,
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 backend=FLOPS_BACKEND.PYTORCH)

        assert (macs, params) == (8, 8)

    def test_input_constructor_dict(self):
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2, bias=True)

            def forward(self, x):
                return self.linear(x)

        def input_constructor(input_res):
            return dict(x=torch.ones(()).new_empty((1, *input_res)))

        macs, params = \
            get_model_complexity_info(CustomLinear(), (3,),
                                      input_constructor=input_constructor,
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend=FLOPS_BACKEND.PYTORCH)

        assert (macs, params) == (8, 8)

    @pytest.mark.parametrize("out_size", [(20, 20), 20])
    def test_func_interpolate_args(self, out_size):
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return nn.functional.interpolate(input=x, size=out_size,
                                                 mode='bilinear', align_corners=False)

        macs, params = \
            get_model_complexity_info(CustomModel(), (3, 10, 10),
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend=FLOPS_BACKEND.PYTORCH)

        assert params == 0
        assert macs == 1200

        CustomModel.forward = lambda self, x: nn.functional.interpolate(x, out_size,
                                                                        mode='bilinear')

        macs, params = \
            get_model_complexity_info(CustomModel(), (3, 10, 10),
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend=FLOPS_BACKEND.PYTORCH)
        assert params == 0
        assert macs == 1200

        CustomModel.forward = lambda self, x: nn.functional.interpolate(x, scale_factor=2,
                                                                        mode='bilinear')

        macs, params = \
            get_model_complexity_info(CustomModel(), (3, 10, 10),
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend=FLOPS_BACKEND.PYTORCH)
        assert params == 0
        assert macs == 1200

    def test_ten_matmul(self, simple_model_mm):
        macs, params = \
            get_model_complexity_info(simple_model_mm, (10, ),
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend=FLOPS_BACKEND.PYTORCH)

        assert params == 0
        assert macs > 0

    def test_aten_ignore(self, simple_model_mm):
        ignored_list = [torch.ops.aten.matmul, torch.ops.aten.mm]
        macs, params = \
            get_model_complexity_info(simple_model_mm, (10, ), backend=FLOPS_BACKEND.ATEN,
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      ignore_modules=ignored_list)

        assert params == 0
        assert macs == 0

    def test_aten_custom(self, simple_model_mm):
        reference = 42
        custom_hooks = {torch.ops.aten.mm: lambda inputs, outputs: reference}

        macs, params = \
            get_model_complexity_info(simple_model_mm, (10, ), backend=FLOPS_BACKEND.ATEN,
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      custom_modules_hooks=custom_hooks)

        assert params == 0
        assert macs == reference

    def test_torch_ignore_func(self, simple_model_mm):
        macs, params = \
            get_model_complexity_info(simple_model_mm, (10, ),
                                      backend=FLOPS_BACKEND.PYTORCH,
                                      as_strings=False,
                                      print_per_layer_stat=False,
                                      backend_specific_config={'count_functional': False})

        assert params == 0
        assert macs == 0
