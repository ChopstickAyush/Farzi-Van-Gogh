import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Erosion2d(nn.Module):

    def __init__(self, m=1):

        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2*m+1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)

        unfolded = self.unfold(x_pad)
        unfolded = unfolded.view(batch_size, c, -1, h, w)
        min_vals, _ = torch.min(unfolded, dim=2)
        output = min_vals.view(batch_size, c, h, w)
        x = torch.where(x != output, output, x)


        # for i in range(c):
        #     channel = self.unfold(x_pad[:, [i], :, :])
        #     channel = torch.min(channel, dim=1, keepdim=True)[0]
        #     channel = channel.view([batch_size, 1, h, w])
        #     x[:, [i], :, :] = channel


        # print((x1 ==x).all())
        return x



class Dilation2d(nn.Module):

    def __init__(self, m=1):

        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2*m+1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)


        unfolded = self.unfold(x_pad)
        unfolded = unfolded.view(batch_size, c, -1, h, w)
        max_vals, _ = torch.max(unfolded, dim=2)
        output = max_vals.view(batch_size, c, h, w)
        x = torch.where(x != output, output, x)


        # for i in range(c):
        #     channel = self.unfold(x_pad[:, [i], :, :])
        #     channel = torch.max(channel, dim=1, keepdim=True)[0]
        #     channel = channel.view([batch_size, 1, h, w])
        #     x[:, [i], :, :] = channel

        # print((x1 == x).all())
        return x
