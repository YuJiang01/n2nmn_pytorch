
from __future__ import absolute_import, division, print_function
from torch.autograd import Variable
from models.layout_assembler import Assembler
from Utils.data_reader import DataReader

import os
import torch
import numpy as np


use_cuda = torch.cuda.is_available()


T_encoder = 45
T_decoder = 20
N = 64
prune_filter_module = True




# Data files
vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
vocab_answer_file = './exp_clevr/data/answers_clevr.txt'


def run_eval(exp_name, snapshot_name, tst_image_set, print_log = False):

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


    if data_reader_tst is not None:
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
        for i, batch in enumerate(data_reader_tst.batches()):

            _, batch_size = batch['input_seq_batch'].shape

            input_text_seq_lens = batch['seq_length_batch']
            input_text_seqs = batch['input_seq_batch']
            input_layouts = batch['gt_layout_batch']

            num_questions_total += batch_size

            n_correct_layout = 0

            input_variable = Variable(torch.LongTensor(input_text_seqs))
            input_variable = input_variable.cuda() if use_cuda else input_variable

            target_variable = Variable(torch.LongTensor(input_layouts))
            target_variable = target_variable.cuda() if use_cuda else target_variable


            myLayouts, myAttentions = model(input_variable, input_text_seq_lens, target_variable)
            predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]

            # compute the accuracy of the predicted layout if ground truth layout exist
            if data_reader_tst.batch_loader.load_gt_layout:
                gt_tokens = batch['gt_layout_batch']
                n_correct_layout += np.sum(np.all(predicted_layouts == gt_tokens, axis=0))

            ##current accuracy
            current_accuracy = n_correct_layout/batch_size
            layout_correct_total += n_correct_layout

            if i % 100 ==0 and print_log:
                print('iter: %d\t accuracy: %f' % (i, current_accuracy))


            # Assemble the layout tokens into network structure
            #expr_list, expr_validity_array = assembler.assemble(predicted_layouts)
            #layout_valid += np.sum(expr_validity_array)

        answer_accuracy = answer_correct_total / num_questions_total
        layout_accuracy = layout_correct_total / num_questions_total
        #layout_validity = layout_valid / num_questions



        return layout_accuracy, layout_correct_total ,num_questions_total


