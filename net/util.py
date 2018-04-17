import numpy as np


def get_params_list(net):
    params = []
    params_num = 0
    for x in net.parameters():
        params.append(x)
        params_num += np.prod(list(x.size()))
    return params,params_num