import torch
import torch.nn as nn

class custom_loss(nn.Module):
    def __init__(self,lambda_entropy=0):
        super(custom_loss, self).__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, neg_entropy, answer_loss, policy_gradient_losses):

        return torch.mean(neg_entropy) * self.lambda_entropy \
               + torch.mean(answer_loss) + torch.mean(policy_gradient_losses)
