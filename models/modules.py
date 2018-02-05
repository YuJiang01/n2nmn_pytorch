
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
    '''
    Mapping image_feat_grid X text_param ->att.grid
    (N,H,W,D_image) X (N,1,D_text) --> [N,H,W,1]
    '''
    def __init__(self,image_dim,text_dim,map_dim):
        super(FindModule,self).__init__()
        self.map_dim = map_dim
        self.conv1 = nn.Conv2d(image_dim,map_dim,kernel_size=1)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)
        self.textfc = nn.Linear(text_dim,map_dim)

    def forward(self, input_image,input_text):
        image_mapped = self.conv1(input_image)
        text_mapped = self.textfc(input_text).view(-1, 1, 1, self.map_dim).expand_as(image_mapped)
        eltwize_mult = image_mapped * text_mapped
        eltwize_mult = F.normalize(eltwize_mult, p=2, dim=3)
        att_grid = self.conv2(eltwize_mult)
        return att_grid



class FilterModule(nn.Module):
    def __init__(self, findModule, andModule):
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