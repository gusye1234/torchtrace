"""
    torchtrace.utils
    store some useful tools function
"""
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from .tracer import tracer
from .base import zeros_like, ones_like#, operate

def load(name):
    with open(name, 'rb') as f:
        weights = pickle.load(f)
    return weights

def save(name, seq_list):
    with open(name, 'wb') as f:
        pickle.dump(seq_list, f)

def load_from_torch(name:str, names_order:dict=None):
    """
        convert state dict into list according to `names_order`
    """
    try:
        state_dict = torch.load(name)
    except RuntimeError:
        state_dict = torch.load(name, map_location=torch.device('cpu'))
    except FileNotFoundError:
        raise FileNotFoundError
    dict_names = list(state_dict.keys())
    if names_order is None:
        names_order = dict_names
    try:
        assert len(dict_names) == len(names_order)
        weights = []
        for name in names_order:
            assert name in dict_names
            weights.append(state_dict[name])
    except AssertionError:
        raise AssertionError(f"expected weights from {names_order}\nbut got {dict_names}")
    return weights

def set_seed(seed):
    '''
        fix Randomness
    '''
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def from_numpy(data : np.ndarray):
    return tracer(data)

def iterate_regions(image, kernel, stride):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape[-2], image.shape[-1]
    for i in range(0, h - kernel[0] + 1, stride[0]):
        for j in range(0, w - kernel[1] + 1, stride[1]):
            if (i+kernel[0] > h) or (j+kernel[1] > w):
                continue
            yield image[...,i:i+kernel[0], j:j+kernel[1]], i//stride[0], j//stride[1]


def inverse_pad_2d(imgs, padding):
    """
        extract matrix from a padded one
    """
    h, w = imgs.shape[-2], imgs.shape[-1]
    return imgs[..., padding[2]:h - padding[3], padding[0]:w-padding[1]]


def generate_grad(imgs, filters, bias, kernel, stride, padding, filter_grad, bias_grad, grad):
    '''
        Design for Conv2d operator
        Generate gradients of convolution layer
    '''
    f_num = filters.shape[0]
    batch = grad.shape[0]
    padding = (padding[1], padding[1], padding[0], padding[0])
    imgs = F.pad(imgs, padding, 'constant', value=0)
    next_grad = zeros_like(imgs)
    for batch_i in range(batch):
        image = imgs[batch_i]
        grad_i = grad[batch_i]
        for (region, i, j) in iterate_regions(image, kernel, stride):
            for f in range(f_num):
                try:
                    filter_grad[f] += grad_i[f, i, j]*region
                    bias_grad[f] += grad_i[f,i,j]
                    next_grad[batch_i,:, i:i+kernel[0], j:j+kernel[1]] += grad_i[f,i,j]*filters[f]
                except:
                    print('----------------------')
                    print(filters.shape, bias_grad.shape)
                    print(image.shape, region.shape, grad_i.shape)
                    print(kernel, stride)
                    print(i,j)
                    raise IndexError
    # next_grad = operate.generate_grad(imgs, filters, bias, kernel, stride, padding, filter_grad, bias_grad, grad)
    # padding = (padding[1], padding[1], padding[0], padding[0])
    return inverse_pad_2d(next_grad, padding)

def refine_para(x):
    '''
        Design for Conv2d
    '''
    if isinstance(x, int):
        x = (x,x)
    try:
        assert len(x) == 2
    except AssertionError:
        raise AssertionError(f'expect to 2-dim, got {x}')
    return x    
    
def arange(length):
    """
        start from 0, wrapper for np.arange
    """
    return torch.Tensor(np.arange(length)).int()


def prRed(skk): print("\033[91m{}\033[00m".format(skk))

def prGreen(skk): print("\033[92m{}\033[00m".format(skk))

def prYellow(skk): print("\033[93m{}\033[00m".format(skk))

def prCyan(skk): print("\033[36m{}\033[00m".format(skk))