import torch
import torch.nn as nn
from collections import OrderedDict

class hourglass_deconv_network(nn.Module):
    def __init__(self):
        super(hourglass_deconv_network, self).__init__()
        ch_in = 3
        ch_pre = ch_in
        self.down_layers_kernels = OrderedDict([("d1", 8), ("d2", 16), ("d3", 32), ("d4", 64), ("d5", 128), ("mid", 256)])
        self.down_kernels_size = 5
        self.up_layers_kernels =  OrderedDict([("u5", 128), ("u4", 64), ("u3", 32), ("u2", 16), ("u1", 8)])
        self.up_kernels_size = 4
        self.skip_layers_kernels =  OrderedDict([("s5", 4), ("s4", 4), ("s3", 4), ("s2", 0), ("s1", 0)])
        self.skip_kernels_size = 5
        # bulid down layers
        for key , value in self.down_layers_kernels.items():
            self._modules[key] = nn.Sequential(
                nn.Conv2d(ch_pre, value, self.down_kernels_size, 2,  2),
                nn.BatchNorm2d(value),
                nn.ReLU())
            ch_pre = value

        # build up layers
        for key, value in self.up_layers_kernels.items():
            if self.skip_layers_kernels['s'+key[-1]] !=0:
                value_without_skip = value - self.skip_layers_kernels['s'+key[-1]]
            else:
                value_without_skip = value
            self._modules[key] = nn.Sequential(
                nn.ConvTranspose2d(ch_pre, value_without_skip, self.up_kernels_size, 2, 1),
                nn.BatchNorm2d(value_without_skip),
                nn.ReLU())
            ch_pre = value

        # build out layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ch_pre, ch_in, self.up_kernels_size, 2, 1),
            nn.BatchNorm2d(ch_in),
            nn.Sigmoid())

        # build skip layers
        for key , value in self.skip_layers_kernels.items():
            if value != 0:
                ch_pre = self.down_layers_kernels['d'+key[-1]]
                self._modules[key] =  nn.Sequential(
                    nn.Conv2d(ch_pre, value, self.skip_kernels_size, 1, 2),
                    nn.BatchNorm2d(value),
                    nn.ReLU())

    def forward(self, input):
        feature = {'input' : input}
        pre_key = 'input'
        for key in self.down_layers_kernels.keys():
            feature[key] = self._modules[key](feature[pre_key])
            pre_key = key
        for key in self.skip_layers_kernels.keys():
            if self.skip_layers_kernels[key] != 0:
                feature[key] = self._modules[key](feature['d'+key[-1]])
        for key in self.up_layers_kernels.keys():
            out = self._modules[key](feature[pre_key])
            if self.skip_layers_kernels['s'+key[-1]] != 0:
                feature[key] = torch.cat([out, feature['s'+key[-1]]], 1)
            else:
                feature[key] = out
            pre_key = key
        out = self.out(feature[pre_key])
        return out
