import torch.nn as nn
import random
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self,inputfeature):
        super(Generator,self).__init__()
        self.input=inputfeature
        self.input_size=inputfeature.size(2)

    def sub_generator(self,n):
        input=self.input
        l = []
        for a in range(n):
            for b in range(n):
                l.append([a, b])
        block_size = self.input_size // n
        rounds = n ** 2
        random.shuffle(l)
        patchs_map = input.clone()
        for i in range(rounds):
            x, y = l[i]
            temp = patchs_map[..., 0:block_size, 0:block_size].clone()
            patchs_map[..., 0:block_size, 0:block_size] = patchs_map[..., x * block_size:(x + 1) * block_size,
                                                       y * block_size:(y + 1) * block_size].clone()
            patchs_map[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

        return patchs_map



