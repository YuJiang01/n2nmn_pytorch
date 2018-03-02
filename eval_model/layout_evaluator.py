from __future__ import absolute_import, division, print_function
from torch.autograd import Variable
from models.layout_assembler import Assembler
from Utils.data_reader import DataReader

import os
import torch
import numpy as np

from global_variables.global_variables import use_cuda


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

    module_snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, "model_"+snapshot_name)


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


        ##load my model
        myModel = torch.load(module_snapshot_file)


        for i, batch in enumerate(data_reader_tst.batches()):

            _, batch_size = batch['input_seq_batch'].shape

            input_text_seq_lens = batch['seq_length_batch']
            input_text_seqs = batch['input_seq_batch']
            input_layouts = batch['gt_layout_batch']
            input_images = batch['image_feat_batch']
            input_answers = batch['answer_label_batch']

            num_questions_total += batch_size


            input_txt_variable = Variable(torch.LongTensor(input_text_seqs))
            input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

            input_layout_variable = None

            _, _, myAnswer, predicted_layouts, expr_validity_array,_ = myModel(
                input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                input_layout_variable=input_layout_variable,
                input_answers=None, input_images=input_images,sample_token=False)


            layout_correct_total += np.sum(predicted_layouts == input_layouts, axis=0)


            answer_correct_total += np.sum(np.logical_and(expr_validity_array, myAnswer == input_answers))

            layout_valid_total += np.sum(expr_validity_array)

            ##current accuracy
            layout_accuracy = layout_correct_total / num_questions_total
            answer_accuracy = answer_correct_total / num_questions_total
            layout_validity = layout_valid_total / num_questions_total

            if (i+1)%100 ==0 and print_log:
                print("iter:", i + 1, " layout_accuracy=%.4f"% layout_accuracy,
                      " answer_accuracy=%.4f"% answer_accuracy,
                      " layout_validity=%.4f"% layout_validity,)









        return layout_accuracy, layout_correct_total ,num_questions_total, answer_accuracy