import math


def kaiming_uniform(tensor, fan_in):
    bound = math.sqrt(6 / fan_in)
    tensor.uniform_(-bound, bound)


def glorot_uniform(tensor, fan_in, fan_out):
    bound = math.sqrt(6 / (fan_in + fan_out))
    tensor.uniform_(-bound, bound)


def lecun_uniform(tensor, fan_in):
    bound = math.sqrt(3 / fan_in)
    tensor.uniform_(-bound, bound)


def zero(tensor):
    tensor.fill_(0.0)


def one(tensor):
    tensor.fill_(1.0)
