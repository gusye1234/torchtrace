"""
    A simple autograd implementation based on PyTorch
"""
from . import tracer
from . import init
from . import optim
from .tracer import tracer
from .nn import Sequential
from .utils import load, save, set_seed, from_numpy
from .utils import prYellow, prGreen, prCyan


__doc__ = """
torchtrace, a simple auto-grad tools designed for Sequential model
"""

__author__ = """
Jianbai Ye
"""