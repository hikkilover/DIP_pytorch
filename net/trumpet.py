import torch
import torch.nn as nn
from superparams import trumpetNetSuperparams


def basic_layer(ch_in, ch_out, kernel_size, stride, mode='conv'):
    if mode == 'conv':
        to_pad =  int((kernel_size - 1)/2)
        return nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=to_pad),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear'))

    
class trumpetNetwork(nn.Module):
    def __init__(self, json, ch_in=3, ch_out=3):
        super(trumpetNetwork, self).__init__()

        self.NSP = trumpetNetSuperparams(json)
        # bulid basic layers
        ch_pre = ch_in
        for key , value in self.NSP.layers_kernels.items():
            self._modules[key] = basic_layer(ch_pre, value, self.NSP.kernels_size, stride=1, mode='conv')
            ch_pre = value

        # build out layer
        self.out =  basic_layer(ch_pre, ch_out, self.NSP.kernels_size, stride=1, mode='conv')

    def forward(self, input):
        feature = {'input' : input}
        pre_key = 'input'
        for key in self.NSP.layers_kernels.keys():
            feature[key] = self._modules[key](feature[pre_key])
            pre_key = key
        out = self.out(feature[pre_key])
        return out 