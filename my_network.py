import torch
import torch.nn as nn
from net_util import hourglassNetSuperparams

def down_layer(ch_in, ch_out, kernel_size, stride, padding, mode='conv'):
    if mode == 'conv':
        return nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU())

def up_layer(ch_in, ch_out, kernel_size, stride, padding, act = 'relu',mode='deconv'):
    if (mode == 'deconv'):
        if (act == 'relu'):
            return nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding),
                   nn.BatchNorm2d(ch_out),
                   nn.ReLU())
        elif (act == 'sigmoid'):
            return nn.Sequential(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding),
                   nn.BatchNorm2d(ch_out),
                   nn.Sigmoid())
    
def skip_layer(ch_in, ch_out, kernel_size, stride, padding, mode='conv'):
    if mode == 'conv':
        return nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU())

class hourglass_network(nn.Module):
    def __init__(self, json, ch_in=3):
        super(hourglass_network, self).__init__()

        self.NSP = hourglassNetSuperparams(json)
        # bulid down layers
        ch_pre = ch_in
        for key , value in self.NSP.down_layers_kernels.items():
            self._modules[key] = down_layer(ch_pre, value, self.NSP.down_kernels_size, 2,  2)
            ch_pre = value

        # build up layers
        for key, value in self.NSP.up_layers_kernels.items():
            if self.NSP.skip_layers_kernels['s'+key[-1]] !=0:
                value_without_skip = value - self.NSP.skip_layers_kernels['s'+key[-1]]
            else:
                value_without_skip = value
            self._modules[key] = up_layer(ch_pre, value_without_skip, self.NSP.up_kernels_size, 2,  1, act = 'relu', mode='deconv')
            ch_pre = value

        # build out layer
        self.out = up_layer(ch_pre, ch_in, self.NSP.up_kernels_size, 2, 1,  act = 'sigmoid', mode='deconv')

        # build skip layers
        for key , value in self.NSP.skip_layers_kernels.items():
            if value != 0:
                ch_pre = self.NSP.down_layers_kernels['d'+key[-1]]
                self._modules[key] = skip_layer(ch_pre, value, self.NSP.skip_kernels_size, 1, 2)

    def forward(self, input):
        feature = {'input' : input}
        pre_key = 'input'
        for key in self.NSP.down_layers_kernels.keys():
            feature[key] = self._modules[key](feature[pre_key])
            pre_key = key
        for key in self.NSP.skip_layers_kernels.keys():
            if self.NSP.skip_layers_kernels[key] != 0:
                feature[key] = self._modules[key](feature['d'+key[-1]])
        for key in self.NSP.up_layers_kernels.keys():
            out = self._modules[key](feature[pre_key])
            if self.NSP.skip_layers_kernels['s'+key[-1]] != 0:
                feature[key] = torch.cat([out, feature['s'+key[-1]]], 1)
            else:
                feature[key] = out
            pre_key = key
        out = self.out(feature[pre_key])
        return out 