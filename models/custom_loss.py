import torch
import torch.nn as nn

class custom_loss(nn.Module):
    def __init__(self,lambda_entropy):
        super(custom_loss, self).__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, neg_entropy, answer_loss, policy_gradient_losses=None,layout_loss =None):
        answer = torch.mean(answer_loss)
        #entropy = torch.mean(neg_entropy)
        #policy_gradient = torch.mean(policy_gradient_losses)
        #print(" answer= %f, entropy  = %f, policy_gradient = %f" %
        #          (answer,entropy,policy_gradient))

        if layout_loss is None:
            return torch.mean(neg_entropy) * self.lambda_entropy +\
               torch.mean(answer_loss)+torch.mean(policy_gradient_losses), answer
        else:
            return answer + layout_loss, answer
