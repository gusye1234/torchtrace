import torch
import numpy as np

def calculate_fan(weight):
    shape = tuple(weight.shape)
    return shape[-1], shape[-2]

def uniform_(weight, bounds=None):
    if bounds is None:
        bounds = [0,1]
    shape = tuple(weight.data.shape)
    weight.data = torch.Tensor(np.random.uniform(*bounds, size=shape))
    return weight
    
def normal_(weight, loc=None,std = None):
    loc = 0. if loc is None else loc
    std = 1. if std is None else std
    
    weight.data = torch.Tensor(np.random.normal(loc=loc, scale=std))
    return weight
    
def kaiming_uniform_(weight, k=None, a = None):
    k = weight.shape[-1] if k is None else k
    a = np.sqrt(5) if a is None else a

    bound = np.sqrt(6./(( 1 + a**2)*k))
    bounds = [-bound, bound]
    return uniform_(weight, bounds=bounds)
    
def kaiming_normal_(weight, k=None, a=None):
    if k is None:
        k = weight.shape[-1]
    if a is None:
        a = 1.
    std = np.sqrt( 2./((1 + a**2) * k))
    return normal_(weight, std=std)
    