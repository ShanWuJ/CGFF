import torch.nn as nn
import random
import numpy as np
import torch

# cross-granularity random swapper
class Swapper(nn.Module):
    def __init__(self):
        super(Swapper,self).__init__()

    def swap(self,x1, x2):
        max = x1.size(2)
        rmin = 2
        rmax = max // 3
        endpoin = max - 1
        n = random.randint(rmin, rmax)
        start = random.randint(0, endpoin)
        out1 = x1.clone()
        out2 = x2.clone()
        if n > endpoin - start:
            patch = out1[..., start - n:start, start - n:start].clone()
            out1[..., start - n:start, start - n:start] = out2[..., start - n:start, start - n:start].clone()
            out2[..., start - n:start, start - n:start] = patch
        else:
            patch = out1[..., start:start + n, start:start + n].clone()
            out1[..., start:start + n, start:start + n] = out2[..., start:start + n, start:start + n].clone()
            out2[..., start:start + n, start:start + n] = patch
        return out1, out2