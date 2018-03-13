import numpy as np

# Module parameters
H_feat = 10
W_feat = 15
D_feat = 512
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 512
num_layers = 2
encoder_dropout = False
decoder_dropout = False
decoder_sampling = True
T_encoder = 45
T_decoder = 10
N = 64
prune_filter_module = True

# Training parameters
invalid_expr_loss = np.log(28)  # loss value when the layout is invalid
lambda_entropy = 0.01
weight_decay = 0
baseline_decay = 0.99
max_grad_l2_norm = 10
max_iter = 120000
snapshot_interval = 10000
learning_rate = 0.001

