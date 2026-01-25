import numpy as np
import pickle
from operator import itemgetter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import datetime

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            weight = (param.data for name, param in m.named_parameters() if "weight" in name)
            for w in weight:
                torch.nn.init.xavier_uniform_(m.weight)

        if isinstance(m, torch.nn.LSTM):
            ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
            for w in ih:
                nn.init.xavier_uniform(w)
            for w in hh:
                nn.init.orthogonal(w)

        if isinstance(m, torch.nn.Conv2d):
            weight = (param.data for name, param in m.named_parameters() if "weight" in name)
            for w in weight:
                torch.nn.init.xavier_uniform_(m.weight)

        # b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        # for w in b:
        #     nn.init.constant(w, 0)

    print(self.__class__.__name__ + ":\n" + str(list(self.modules())[0]))
