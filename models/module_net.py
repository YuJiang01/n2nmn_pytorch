import torch
import torch.nn as nn
import torch.nn.functional as F


class module_net(nn.Module):
    def __init__(self,in_image_dim,in_text_dim, output_dim):
        super(module_net,self).__init__()
        self.in_image_dim = in_image_dim
        self.in_text_dim = in_text_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120,self.output_dim)

    def forward(self, input_image,input_text, expr_list):
        input_image = self.pool(F.relu(self.conv1(input_image)))
        input_image = input_image.view(-1, 16 * 5 * 5)
        input_image = F.relu(self.fc1(input_image))
        input_image = self.fc2(input_image)

        return input_image


