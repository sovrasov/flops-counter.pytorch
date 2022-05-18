import pytest
import torch.nn as nn

from ptflops import get_model_complexity_info


class TestOperations:
    @pytest.fixture
    def default_input_image_size(self):
        return (3, 224, 224)

    def test_conv(self, default_input_image_size):
        net = nn.Sequential(nn.Conv2d(3, 2, 3, bias=True))
        macs, params = get_model_complexity_info(net, default_input_image_size,
                                                 as_strings=False,
                                                 print_per_layer_stat=False)

        assert params == 3 * 3 * 2 * 3 + 2
        assert int(macs) == 2759904

    def test_fc(self):
        net = nn.Sequential(nn.Linear(3, 2, bias=True))
        macs, params = get_model_complexity_info(net, (3,),
                                                 as_strings=False,
                                                 print_per_layer_stat=False)

        assert params == 3 * 2 + 2
        assert int(macs) == 8
