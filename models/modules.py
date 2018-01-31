
import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneModule(nn.Module):

    def __init__(self):
        super(SceneModule,self).__init__()

    def forward(self, x):
        grid = torch.ones(x.size())
        return grid


class FindModule(nn.Module):
    def __init__(self):
        super(FindModule,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(1, 10, kernel_size=5)


class FilterModule(nn.Module):
    def __init__(self):
        super(FilterModule,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(1, 10, kernel_size=5)


class FindSamePropertyModule(nn.Module):
    def __init__(self):
        super(FindSamePropertyModule,self).__init__()


class TransformModule(nn.Module):
    def __init__(self):
        super(TransformModule,self).__init__()

class AndModule(nn.Module):
    def __init__(self):
        super(AndModule,self).__init__()

class OrModule(nn.Module):
    def __init__(self):
        super(OrModule,self).__init__()


class CountModule(nn.Module):
    def __init__(self):
        super(CountModule,self).__init__()


class ExistModule(nn.Module):
    def __init__(self):
        super(ExistModule,self).__init__()


class EqualNumModule(nn.Module):
    def __init__(self):
        super(EqualNumModule,self).__init__()

class MoreNumModule(nn.Module):
    def __init__(self):
        super(MoreNumModule,self).__init__()

class LessNumModule(nn.Module):
    def __init__(self):
        super(LessNumModule,self).__init__()

class SamePropertyModule(nn.Module):
    def __init__(self):
        super(SamePropertyModule,self).__init__()

class DescribeModule(nn.Module):
    def __init__(self):
        super(DescribeModule,self).__init__()