import torch
import torch.nn as nn
import math

import torch.nn.functional as F

import time
import numpy as np

from utils.options import args

device = torch.device(f"cuda:{args.gpus[0]}")


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      #grad_input = grad_output
      return grad_output

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, bitW):
    super(weight_quantize_fn, self).__init__()
    #assert bitW <= 8 or bitW == 32
    self.bitW = bitW
    self.uniform_q = uniform_quantize(k=bitW)

  def forward(self, x):
    if self.bitW == 1:
      E = torch.mean(torch.abs(x)).detach()
      x = self.uniform_q(x / E) * E
    else:
      x = self.uniform_q(x)
    return x


class activation_quantize_fn(nn.Module):
  def __init__(self, abitW):
    super(activation_quantize_fn, self).__init__()
    #assert abitW <= 8 or abitW == 32
    self.abitW = abitW
    self.uniform_q = uniform_quantize(k=abitW)

  def forward(self, x):
    if self.abitW == 32:
      return x
    else:
      m = x.min()
      M = x.max()
    
      x = self.uniform_q((x-m) / (M-m))
      # print(np.unique(activation_q.detach().numpy()))

      return self.uniform_q(x) *(M-m) + m


def Quant_conv2d(bitW):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.bitW = bitW
      self.quantize_fn = weight_quantize_fn(bitW=bitW)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

def Quant_Linear(bitW):
    class Linear_Q(nn.Linear):
        """
        Class to quantize given linear layer weights
        """
        def __init__(self, in_features, out_features):
            """
            weight: bit-setting for weight
            full_precision_flag: full precision or not
            running_stat: determines whether the activation range is updated or froze
            """
            super(Linear_Q, self).__init__(in_features, out_features)
            self.bitW = bitW
            self.quantize_fn = weight_quantize_fn(bitW=bitW)

            self.in_features = in_features
            self.out_features = out_features
   
            
        def forward(self, x):
            """
            using quantized weights to forward activation x
            """
            weight_q = self.quantize_fn(self.weight)
            return F.linear(x, weight=weight_q, bias=self.bias)
        
    return Linear_Q