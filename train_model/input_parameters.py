from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
from global_variables.global_variables import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--exp_name",type=str, default="clevr_gt_layout")
parser.add_argument("--model_type",type=str, choices=[model_type_scratch, model_type_gt, model_type_gt_rl],
                    required=True, help='models:'+ model_type_scratch + ',' +model_type_gt+', '+model_type_gt_rl)
parser.add_argument("--model_path",type=str, required=False)
parser.add_argument("--data_dir",type=str,default="./exp_clevr/data")
parser.add_argument("--image_feat_dir",type=str,default="/Users/tinayujiang/work/clevr_dataset/data/vgg_pool5/train")
parser.add_argument("--out_dir",type=str,default="./exp_clevr")
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
exp_name = args.exp_name
model_type = args.model_type



os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


out_dir = args.out_dir
data_dir = args.data_dir
image_feat_dir = args.image_feat_dir

if model_type == model_type_scratch:
    from train_model.from_scratch_hyperparameters import *
elif model_type == model_type_gt:
    from train_model.gt_hyperparameters import *
elif model_type == model_type_gt_rl:
    from train_model.gt_rl_hyperparameters import *
    if(args.model_path == None):
        exit("model ",model_type_gt_rl," require a pretrained model using --model_path")
    model_path = args.model_path
else:
    sys.exit("unknown model type", model_type)



# Log params
#log_dir = './exp_clevr/tb/%s/' % exp_name
log_dir = os.path.join(out_dir, 'tb', exp_name)

# Data files
#vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
#vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
#vocab_answer_file = './exp_clevr/data/answers_clevr.txt'

vocab_question_file = os.path.join(data_dir, 'vocabulary_clevr.txt')
vocab_layout_file = os.path.join(data_dir, 'vocabulary_layout.txt')
vocab_answer_file = os.path.join(data_dir, 'answers_clevr.txt')


#imdb_file_trn = './exp_clevr/data/imdb/imdb_trn.npy'
#imdb_file_tst = './exp_clevr/data/imdb/imdb_val.npy'
imdb_file_trn = os.path.join(data_dir, 'imdb/imdb_trn.npy')
imdb_file_tst = os.path.join(data_dir, 'imdb/imdb_val.npy')
image_feat_dir = image_feat_dir

##snapshot directory name
#snapshot_dir = './exp_clevr/tfmodel/%s/' % exp_name

snapshot_dir = os.path.join(out_dir,"tfmodel",exp_name)