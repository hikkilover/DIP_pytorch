import json
from collections import OrderedDict


class netSuperparams(object):
    def __init__(self):
        self.name = 'Default Network'


hnet_params_set = set(['down_layers_kernels', 'down_mode', 'down_kernels_size',
                       'up_layers_kernels', 'up_mode', 'up_kernels_size',
                       'skip_layers_kernels', 'skip_mode', 'skip_kernels_size',
                       'name'])


class hourglassNetSuperparams(netSuperparams):
    def __init__(self, jsonFile):
        super(hourglassNetSuperparams, self).__init__()

        with open(jsonFile) as jf:
            rowDict = json.load(jf, object_pairs_hook=OrderedDict)
        assert set(rowDict.keys()) == hnet_params_set
        for key in rowDict.keys():
            self.__dict__[key] = rowDict[key]
        # assert len(self.down_layers_kernels) - 1 == len(self.up_layers_kernels)
        assert len(self.skip_layers_kernels) == len(self.up_layers_kernels)
        self.depth = len(self.skip_layers_kernels)