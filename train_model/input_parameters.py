from __future__ import absolute_import, division, print_function
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use


os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# Module parameters
H_feat = 10
W_feat = 15
D_feat = 512
embed_dim_txt = 300
embed_dim_nmn = 300
hidden_size = 512
num_layers = 2
encoder_dropout = False
decoder_dropout = False
decoder_sampling = True
T_encoder = 45
T_decoder = 10
N = 64
prune_filter_module = True

# Training parameters
weight_decay = 5e-6
baseline_decay = 0.99
max_grad_l2_norm = 10
max_iter = 80000
snapshot_interval = 10000
exp_name = "clevr_gt_layout"
snapshot_dir = './exp_clevr/tfmodel/%s/' % exp_name

# Log params
log_interval = 100
log_dir = './exp_clevr/tb/%s/' % exp_name

# Data files
vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
vocab_answer_file = './exp_clevr/data/answers_clevr.txt'

imdb_file_trn = './exp_clevr/data/imdb/imdb_trn.npy'
imdb_file_tst = './exp_clevr/data/imdb/imdb_val.npy'


##snapshot directory name
snapshot_dir = './exp_clevr/tfmodel/%s/' % exp_name

