from __future__ import absolute_import, division, print_function
from torch.autograd import Variable
from models.layout_assembler import Assembler
from Utils.data_reader import DataReader
import sys

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from Utils.text_processing import *

from eval_model.layout_evaluator import run_eval


# Data files
vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
vocab_answer_file = './exp_clevr/data/answers_clevr.txt'


exp_name = 'clevr_gt_layout'
tst_image_set = 'val'
out_file = "layout_learning_on_eval_dataset.txt"

snapshot_name='%08d' % 500

T_encoder = 45
T_decoder = 20
N = 64
prune_filter_module = True



imdb_file_tst = './exp_clevr/data/imdb/imdb_%s.npy' % tst_image_set
snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, snapshot_name)

save_file = './exp_clevr/results/%s/%s.%s.txt' % (exp_name, snapshot_name, tst_image_set)
os.makedirs(os.path.dirname(save_file), exist_ok=True)

eval_output_file = './exp_clevr/eval_outputs/%s/%s.%s.txt' % (exp_name, snapshot_name, tst_image_set)
os.makedirs(os.path.dirname(eval_output_file), exist_ok=True)

assembler = Assembler(vocab_layout_file)
data_reader_tst = DataReader(imdb_file_tst, shuffle=False, one_pass=True,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file,
                             prune_filter_module=prune_filter_module)

print('Running test ...')
answer_correct_total = 0
layout_correct_total = 0
layout_valid_total = 0
num_questions_total = 0
answer_word_list = data_reader_tst.batch_loader.answer_dict.word_list
output_answers = []

##load my model
model = torch.load(snapshot_file)

n_total = 0

batch = data_reader_tst.prefetch_queue.get(block=True)
_, batch_size = batch['input_seq_batch'].shape

input_text_seq_lens = batch['seq_length_batch']
input_text_seqs = batch['input_seq_batch']
input_layouts = batch['gt_layout_batch']

num_questions_total += batch_size

n_correct_layout = 0

input_variable = Variable(torch.LongTensor(input_text_seqs))

target_variable = Variable(torch.LongTensor(input_layouts))

myLayouts, myAttentions = model(input_variable, input_text_seq_lens, target_variable)
predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]


sample_idx = 17



def plot_sample(sample_idx):
    example_text = input_text_seqs[:,sample_idx]
    example_att = myAttentions.data.cpu().numpy()[sample_idx,:,:]
    example_layout = predicted_layouts[:,sample_idx]
    word_list = load_str_list(vocab_question_file)
    layout_list=load_str_list(vocab_layout_file)
    sentence = list(map(lambda x:word_list[x],example_text))
    layout = list(map(lambda x: layout_list[x],example_layout))
    sentence_len = sum(1 for i in sentence if i != ';')
    layout_len = sum(1 for i in layout if i != '<eos>')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(example_att[0:layout_len,0:sentence_len],cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+sentence[0:sentence_len],rotation=90)
    ax.set_yticklabels(['']+layout[0:layout_len])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


plot_sample(17)