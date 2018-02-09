
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


'''
NOTE: in all modules, 
image_feat [N,D_image,H,W]
text [N,D_text]
attention [N,1,H,W]
'''



class SceneModule(nn.Module):
    def __init__(self):
        super(SceneModule,self).__init__()

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        N, _, H, W = input_image_feat.shape
        res = torch.ones((N, 1, H, W))
        att_grid = Variable(res)
        att_grid = att_grid.cuda() if use_cuda else att_grid
        return att_grid


class FindModule(nn.Module):
    '''
    Mapping image_feat_grid X text_param ->att.grid
    (N,D_image,H,W) X (N,1,D_text) --> [N,1,H,W]
    '''
    def __init__(self, image_dim, text_dim, map_dim):
        super(FindModule,self).__init__()
        self.map_dim = map_dim
        self.conv1 = nn.Conv2d(image_dim,map_dim,kernel_size=1)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)
        self.textfc = nn.Linear(text_dim,map_dim)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        image_mapped = self.conv1(input_image_feat)  #(N, map_dim, H, W)
        text_mapped = self.textfc(input_text).view(-1, self.map_dim,1,1).expand_as(image_mapped)
        elmtwize_mult = image_mapped * text_mapped
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1) #(N, map_dim, H, W)
        att_grid = self.conv2(elmtwize_mult) #(N, 1, H, W)
        return att_grid



class FilterModule(nn.Module):
    def __init__(self, findModule, andModule):
        super(FilterModule,self).__init__()
        self.andModule = andModule
        self.findModule = findModule

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        find_result = self.findModule(input_image_feat,input_text,input_image_attention1,input_image_attention2)
        att_grid = self.andModule(input_image_feat,input_text,input_image_attention1,find_result)
        return att_grid


class FindSamePropertyModule(nn.Module):
    def __init__(self,output_num_choice, image_dim, text_dim, map_dim):
        super(FindSamePropertyModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.image_dim = image_dim
        self.map_dim = map_dim
        self.text_fc = nn.Linear(text_dim, map_dim)
        self.att_fc_1 = nn.Linear(image_dim, map_dim)
        self.conv1 = nn.Conv2d(image_dim, map_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att_softmax_1 = F.softmax(input_image_attention1.view(-1, H * W),dim=1).view(-1, 1, H*W)
        image_reshape = input_image_feat.view(-1,self.image_dim,H * W)
        att_feat_1 = torch.sum(att_softmax_1 * image_reshape, dim=2)    #[N, image_dim]
        att_feat_1_mapped = self.att_fc_1(att_feat_1).view(-1, self.map_dim,1,1)       #[N, map_dim,1,1]

        text_mapped = self.text_fc(input_text).view(-1,self.map_dim,1,1)

        image_mapped = self.conv1(input_image_feat)  # (N, map_dim, H, W)

        elmtwize_mult = image_mapped * text_mapped * att_feat_1_mapped #[N, map_dim, H, W]
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)

        att_grid = self.conv2(elmtwize_mult)

        return att_grid


class TransformModule(nn.Module):
    def __init__(self, image_dim, text_dim, map_dim,kernel_size=5, padding=2):
        super(TransformModule,self).__init__()
        self.map_dim = map_dim
        self.conv1 = nn.Conv2d(1, map_dim, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(map_dim, 1, kernel_size=1)
        self.textfc = nn.Linear(text_dim,map_dim)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        image_att_mapped = self.conv1(input_image_attention1)  #(N, map_dim, H, W)
        text_mapped = self.textfc(input_text).view(-1, self.map_dim,1,1).expand_as(image_att_mapped)
        elmtwize_mult = image_att_mapped * text_mapped
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1) #(N, map_dim, H, W)
        att_grid = self.conv2(elmtwize_mult) #(N, 1, H, W)
        return att_grid


class AndModule(nn.Module):
    def __init__(self):
        super(AndModule,self).__init__()

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        return torch.max(input_image_attention1, input_image_attention2)


class OrModule(nn.Module):
    def __init__(self):
        super(OrModule,self).__init__()
    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        return torch.min(input_image_attention1, input_image_attention2)



class CountModule(nn.Module):
    def __init__(self,output_num_choice, image_height, image_width):
        super(CountModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height*image_width + 3, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att_all = input_image_attention1.view(-1, H*W) ##flatten attention to [N, H*W]
        att_avg = torch.mean(att_all, 1, keepdim=True)
        att_min = torch.min(att_all, 1, keepdim=True)[0]
        att_max = torch.max(att_all,1, keepdim=True)[0]
        att_concat = torch.cat((att_all, att_avg, att_min, att_max), 1)
        scores = self.lc_out(att_concat)
        return scores




class ExistModule(nn.Module):
    def __init__(self,output_num_choice, image_height, image_width):
        super(ExistModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height*image_width + 3, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att_all = input_image_attention1.view(-1, H*W) ##flatten attention to [N, H*W]
        att_avg = torch.mean(att_all, 1, keepdim=True)
        att_min = torch.min(att_all, 1, keepdim=True)[0]
        att_max = torch.max(att_all, 1, keepdim=True)[0]
        att_concat = torch.cat((att_all, att_avg, att_min, att_max), 1)
        scores = self.lc_out(att_concat)
        return scores


class EqualNumModule(nn.Module):
    def __init__(self,output_num_choice, image_height, image_width):
        super(EqualNumModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height*image_width *2 + 6, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att1_all = input_image_attention1.view(-1, H * W) ##flatten attention to [N, H*W]
        att1_avg = torch.mean(att1_all, 1, keepdim=True)
        att1_min = torch.min(att1_all, 1, keepdim=True)[0]
        att1_max = torch.max(att1_all, 1, keepdim=True)[0]

        att2_all = input_image_attention2.view(-1, H * W)  ##flatten attention to [N, H*W]
        att2_avg = torch.mean(att2_all, 1, keepdim=True)
        att2_min = torch.min(att2_all, 1, keepdim=True)[0]
        att2_max = torch.max(att2_all, 1, keepdim=True)[0]

        att_concat = torch.cat((att1_all, att1_avg, att1_min, att1_max,att2_all, att2_avg, att2_min, att2_max), 1)
        scores = self.lc_out(att_concat)
        return scores

class MoreNumModule(nn.Module):
    def __init__(self, output_num_choice, image_height, image_width):
        super(MoreNumModule, self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height * image_width * 2 + 6, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att1_all = input_image_attention1.view(-1, H * W)  ##flatten attention to [N, H*W]
        att1_avg = torch.mean(att1_all, 1, keepdim=True)
        att1_min = torch.min(att1_all, 1, keepdim=True)[0]
        att1_max = torch.max(att1_all, 1, keepdim=True)[0]

        att2_all = input_image_attention2.view(-1, H * W)  ##flatten attention to [N, H*W]
        att2_avg = torch.mean(att2_all, 1, keepdim=True)
        att2_min = torch.min(att2_all, 1, keepdim=True)[0]
        att2_max = torch.max(att2_all, 1, keepdim=True)[0]

        att_concat = torch.cat((att1_all, att1_avg, att1_min, att1_max, att2_all, att2_avg, att2_min, att2_max), 1)
        scores = self.lc_out(att_concat)
        return scores

class LessNumModule(nn.Module):
    def __init__(self, output_num_choice, image_height, image_width):
        super(LessNumModule, self).__init__()
        self.out_num_choice = output_num_choice
        self.lc_out = nn.Linear(image_height * image_width * 2 + 6, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att1_all = input_image_attention1.view(-1, H * W)  ##flatten attention to [N, H*W]
        att1_avg = torch.mean(att1_all, 1, keepdim=True)
        att1_min = torch.min(att1_all, 1, keepdim=True)[0]
        att1_max = torch.max(att1_all, 1, keepdim=True)[0]

        att2_all = input_image_attention2.view(-1, H * W)  ##flatten attention to [N, H*W]
        att2_avg = torch.mean(att2_all, 1, keepdim=True)
        att2_min = torch.min(att2_all, 1, keepdim=True)[0]
        att2_max = torch.max(att2_all, 1, keepdim=True)[0]

        att_concat = torch.cat((att1_all, att1_avg, att1_min, att1_max, att2_all, att2_avg, att2_min, att2_max), 1)
        scores = self.lc_out(att_concat)
        return scores

class SamePropertyModule(nn.Module):
    def __init__(self,output_num_choice, image_dim, text_dim, map_dim):
        super(SamePropertyModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.image_dim = image_dim
        self.text_fc = nn.Linear(text_dim, map_dim)
        self.att_fc_1 = nn.Linear(image_dim, map_dim)
        self.att_fc_2 = nn.Linear(image_dim, map_dim)
        self.lc_out = nn.Linear(map_dim, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att_softmax_1 = F.softmax(input_image_attention1.view(-1, H * W),dim=1).view(-1, 1, H*W)
        att_softmax_2 = F.softmax(input_image_attention2.view(-1, H * W), dim=1).view(-1, 1, H*W)
        image_reshape = input_image_feat.view(-1,self.image_dim,H * W)
        att_feat_1 = torch.sum(att_softmax_1 * image_reshape, dim=2)    #[N, image_dim]
        att_feat_2 = torch.sum(att_softmax_2 * image_reshape, dim=2)
        att_feat_1_mapped = self.att_fc_1(att_feat_1)       #[N, map_dim]
        att_feat_2_mapped = self.att_fc_2(att_feat_2)

        text_mapped = self.text_fc(input_text)
        elmtwize_mult = att_feat_1_mapped * text_mapped * att_feat_2_mapped #[N, map_dim]
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)
        scores = self.lc_out(elmtwize_mult)

        return scores

class DescribeModule(nn.Module):
    def __init__(self,output_num_choice, image_dim, text_dim, map_dim):
        super(DescribeModule,self).__init__()
        self.out_num_choice = output_num_choice
        self.image_dim = image_dim
        self.text_fc = nn.Linear(text_dim, map_dim)
        self.att_fc_1 = nn.Linear(image_dim, map_dim)
        self.lc_out = nn.Linear(map_dim, self.out_num_choice)

    def forward(self, input_image_feat, input_text, input_image_attention1=None, input_image_attention2=None):
        H, W = input_image_attention1.shape[2:4]
        att_softmax_1 = F.softmax(input_image_attention1.view(-1, H * W),dim=1).view(-1, 1, H*W)
        image_reshape = input_image_feat.view(-1,self.image_dim,H * W) #[N,image_dim,H*W]
        att_feat_1 = torch.sum(att_softmax_1 * image_reshape, dim=2)    #[N, image_dim]
        att_feat_1_mapped = self.att_fc_1(att_feat_1)       #[N, map_dim]

        text_mapped = self.text_fc(input_text)
        elmtwize_mult = att_feat_1_mapped * text_mapped  #[N, map_dim]
        elmtwize_mult = F.normalize(elmtwize_mult, p=2, dim=1)
        scores = self.lc_out(elmtwize_mult)

        return scores