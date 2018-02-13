import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()



class module_net(nn.Module):

    ##initiate all small modules which will be used here
    def __init__(self, image_height, image_width, in_image_dim, in_text_dim, out_num_choices, map_dim):
        super(module_net,self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.in_image_dim = in_image_dim
        self.in_text_dim = in_text_dim
        self.out_num_choices = out_num_choices
        self.map_dim = map_dim
        self.SceneModule = SceneModule()
        self.FindModule = FindModule(image_dim=in_image_dim, text_dim=in_text_dim, map_dim= map_dim)
        self.TransformModule = TransformModule(image_dim=in_image_dim, text_dim=in_text_dim, map_dim = map_dim)
        self.AndModule = AndModule()
        self.OrModule = OrModule()
        self.FilterModule = FilterModule(findModule=self.FindModule, andModule=self.AndModule)
        self.FindSamePropertyModule = FindSamePropertyModule(
            output_num_choice=out_num_choices,image_dim=in_image_dim, text_dim=in_text_dim, map_dim = map_dim)

        self.CountModule = CountModule(output_num_choice=out_num_choices,
                                       image_height=image_height, image_width= image_width)

        self.ExistModule = ExistModule(output_num_choice=out_num_choices,
                                       image_height=image_height, image_width= image_width)

        self.EqualNumModule = EqualNumModule(output_num_choice=out_num_choices,
                                             image_height=image_height, image_width= image_width)

        self.MoreNumModule = MoreNumModule(output_num_choice=out_num_choices,
                                           image_height=image_height, image_width= image_width)

        self.LessNumModule = LessNumModule(output_num_choice=out_num_choices,
                                           image_height=image_height, image_width= image_width)

        self.SamePropertyModule = SamePropertyModule(
            output_num_choice=out_num_choices,image_dim=in_image_dim,text_dim=in_text_dim, map_dim=map_dim)

        self.DescribeModule = DescribeModule(
            output_num_choice=out_num_choices,image_dim=in_image_dim, text_dim=in_text_dim, map_dim = map_dim)

        self.layout2module = {
            '_Filter': self.FilterModule,
            '_FindSameProperty': self.FindSamePropertyModule,
            '_Transform': self.TransformModule,
            '_And': self.AndModule,
            '_Or': self.OrModule,
            '_Count': self.CountModule,
            '_Exist': self.ExistModule,
            '_EqualNum': self.EqualNumModule,
            '_MoreNum': self.MoreNumModule,
            '_LessNum': self.LessNumModule,
            '_SameProperty': self.SamePropertyModule,
            '_Describe': self.DescribeModule,
            '_Find': self.FindModule,
            '_Scene': self.SceneModule
        }

    #text[N, D_text]

    def recursively_assemble_network(self,input_image_variable, input_text_attention_variable,expr_list):
        current_module = self.layout2module[expr_list['module']]
        time_idx = expr_list['time_idx']
        text_index = Variable(torch.LongTensor([time_idx]))
        text_index = text_index.cuda() if use_cuda else text_index
        text_at_time = torch.index_select(input_text_attention_variable, dim=1,
                                          index=text_index).view(-1, self.in_text_dim)

        input_0 = None
        input_1 = None

        if 'input_0' in expr_list:
            input_0 = self.recursively_assemble_network(input_image_variable,
                                                        input_text_attention_variable, expr_list['input_0'])
        if 'input_1' in expr_list:
            input_1 = self.recursively_assemble_network(input_image_variable, input_text_attention_variable,
                                                        expr_list['input_1'])

        res = current_module(input_image_variable, text_at_time, input_0, input_1)
        return res


    def forward(self, input_image_variable, input_text_attention_variable, target_answer_variable, expr_list):


        ##for now assume batch_size = 1
        result = self.recursively_assemble_network(input_image_variable,input_text_attention_variable,expr_list)

        return result


