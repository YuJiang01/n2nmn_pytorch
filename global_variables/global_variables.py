import torch


use_cuda = torch.cuda.is_available()

model_type_gt = "gt_layout"
model_type_scratch = "scratch"
model_type_gt_rl = "gt+rl"
