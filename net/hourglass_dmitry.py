import torch
import torch.nn as nn
from superparams import hourglassNetSuperparams
# padding = (kernel_size - 1)/2


def down_layer(ch_in, ch_out, kernel_size, stride, mode='conv'):
    to_pad = int((kernel_size - 1)/2)
    if mode == 'conv':
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.ReLU())

    if mode == 'dmitry':
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, 2, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size, 1, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU())


def up_layer(ch_in, ch_out, kernel_size, stride, mode='deconv'):
    if mode == 'deconv':
        to_pad = (kernel_size - stride)/2
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.ReLU())

    elif mode == 'dmitry':
        to_pad = (kernel_size - 1)/2
        return nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size, 1, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
            nn.Conv2d(ch_out, ch_out, 1, 1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'))

def skip_layer(ch_in, ch_out, kernel_size, stride, mode='conv'):
    to_pad = (kernel_size - 1)/2
    if mode == 'conv':
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.ReLU())

    elif mode == 'dmitry':
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, to_pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU())

def output_layer(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 1, 1, padding=0),
        nn.BatchNorm2d(ch_out),
        nn.Sigmoid())

class hourglassNetwork(nn.Module):
    def __init__(self, json, ch_in=3, ch_out=3):
        super(hourglassNetwork, self).__init__()

        self.NSP = hourglassNetSuperparams(json)
        # bulid down layers
        ch_pre = ch_in
        for key , value in self.NSP.down_layers_kernels.items():
            self._modules[key] = down_layer(ch_pre, value, self.NSP.down_kernels_size, stride = 2,  mode=self.NSP.down_mode)
            ch_pre = value

        # build up layers
        for key, value in self.NSP.up_layers_kernels.items():
            if self.NSP.skip_layers_kernels['s'+key[-1]] !=0:
                value_with_skip = ch_pre + self.NSP.skip_layers_kernels['s'+key[-1]]
            else:
                value_with_skip = ch_pre
            self._modules[key] = up_layer(value_with_skip, value, self.NSP.up_kernels_size, stride = 2, mode=self.NSP.up_mode)
            ch_pre = value

        # build out layer
        self.out = output_layer(ch_pre, ch_out)

        # build skip layers
        for key , value in self.NSP.skip_layers_kernels.items():
            if value != 0:
                ch_pre = self.NSP.down_layers_kernels['d'+key[-1]]
                self._modules[key] = skip_layer(ch_pre, value, self.NSP.skip_kernels_size, stride=1, mode=self.NSP.skip_mode)

    def forward(self, input, print_size=False):
        feature = {'input' : input}
        pre_key = 'input'
        for key in self.NSP.down_layers_kernels.keys():
            feature[key] = self._modules[key](feature[pre_key])
#            if print_size:
#                print(key+' : '+str(feature[key].size()))
            pre_key = key
        for key in self.NSP.skip_layers_kernels.keys():
            if self.NSP.skip_layers_kernels[key] != 0:
                feature[key] = self._modules[key](feature['d'+key[-1]])
#                if print_size:
#                    print(key+' : '+str(feature[key].size()))
        for key in self.NSP.up_layers_kernels.keys():
            if self.NSP.skip_layers_kernels['s'+key[-1]] != 0:
                union_input = torch.cat([feature[pre_key], feature['s'+key[-1]]], 1)
            else:
                union_input = feature[pre_key]
            feature[key] = self._modules[key](union_input)
#            if print_size:
#                print(key+' : '+str(feature[key].size()))                      
            pre_key = key
        out = self.out(feature[pre_key])

        return out 