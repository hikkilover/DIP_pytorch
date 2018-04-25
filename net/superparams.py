import json
from collections import OrderedDict


class netSuperparams(object):
    def __init__(self, name, json_file, params_set):
        self.name = name
        with open(json_file) as jf:
            rowDict = json.load(jf, object_pairs_hook=OrderedDict)
        assert set(rowDict.keys()) == params_set
        for key in rowDict.keys():
            self.__dict__[key] = rowDict[key]
            
            
hnet_params_set = set(['down_layers_kernels', 'down_mode', 'down_kernels_size',
                       'up_layers_kernels', 'up_mode', 'up_kernels_size',
                       'skip_layers_kernels', 'skip_mode', 'skip_kernels_size',
                       'name'])


class hourglassNetSuperparams(netSuperparams):
    def __init__(self, json_file, name='Hourglass Network'):
        super(hourglassNetSuperparams, self).__init__(name, json_file, hnet_params_set)
        assert len(self.skip_layers_kernels) == len(self.up_layers_kernels)
        self.depth = len(self.skip_layers_kernels)


tnet_params_set = set(['layers_kernels', 'mode', 'kernels_size'])


class trumpetNetSuperparams(netSuperparams):
    def __init__(self, json_file, name='Horn Network'):
        super(trumpetNetSuperparams, self).__init__(name, json_file, tnet_params_set)
        self.depth = len(self.layers_kernels)