import math
from ..tensor import tensor


def kaiming_uniform(tensor, fan_in, **kwargs):
    bound = math.sqrt(6 / fan_in)
    tensor.uniform_(-bound, bound)


def glorot_uniform(tensor, fan_in, fan_out):
    bound = math.sqrt(6 / (fan_in + fan_out))
    tensor.uniform_(-bound, bound)


def lecun_uniform(tensor, fan_in, **kwargs):
    bound = math.sqrt(3 / fan_in)
    tensor.uniform_(-bound, bound)


def zero(tensor, **kwargs):
    tensor.fill_(0.0)


def one(tensor, **kwargs):
    tensor.fill_(1.0)
