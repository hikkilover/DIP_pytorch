import torch
from torch.autograd import Variable


def create_random_input(size, mean=0, xigma=1):
    return Variable(torch.add(torch.mul(torch.randn(size).cuda(), xigma), mean))