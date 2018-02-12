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
    layout_snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, "seq2seq_"+snapshot_name)
    module_snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, "module_"+snapshot_name)


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
        myseq2seq = torch.load(layout_snapshot_file)
        myModuleNet = torch.load(module_snapshot_file)

        n_total = 0
        n_correct_answer = 0
        for i, batch in enumerate(data_reader_tst.batches()):

            _, batch_size = batch['input_seq_batch'].shape

            input_text_seq_lens = batch['seq_length_batch']
            input_text_seqs = batch['input_seq_batch']
            input_layouts = batch['gt_layout_batch']
            input_images = batch['image_feat_batch']
            input_answers = batch['answer_label_batch']

            num_questions_total += batch_size

            n_correct_layout = 0

            input_variable = Variable(torch.LongTensor(input_text_seqs))
            input_variable = input_variable.cuda() if use_cuda else input_variable

            target_variable = Variable(torch.LongTensor(input_layouts))
            target_variable = target_variable.cuda() if use_cuda else target_variable


            myLayouts, myAttentions = myseq2seq(input_variable, input_text_seq_lens, target_variable)
            predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]

            # compute the accuracy of the predicted layout if ground truth layout exist
            if data_reader_tst.batch_loader.load_gt_layout:
                gt_tokens = batch['gt_layout_batch']
                n_correct_layout += np.sum(np.all(predicted_layouts == gt_tokens, axis=0))

            ##current accuracy
            current_accuracy = n_correct_layout/batch_size
            layout_correct_total += n_correct_layout

            predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]
            n_correct_layout += np.sum(np.all(predicted_layouts == input_layouts, axis=0))

            ## get the layout information
            expr_list, expr_validity_array = assembler.assemble(predicted_layouts)

            input_answers_variable = Variable(torch.LongTensor(input_answers))
            input_answers_variable = input_answers_variable.cuda() if use_cuda else input_answers_variable

            input_images_variable = Variable(torch.FloatTensor(input_images))
            input_images_variable = input_images_variable.cuda() if use_cuda else input_images_variable

            ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
            input_images_variable = input_images_variable.permute(0, 3, 1, 2)

            answer_loss = 0
            ## for the first step, train sample one by one
            for i_sample in range(batch_size):

                ##skip the case when expr is not validity
                if expr_validity_array[i_sample]:
                    # print("iter:", i_iter, " isample:" ,i_sample)
                    ith_answer_variable = input_answers_variable[i_sample]
                    ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable
                    text_index = Variable(torch.LongTensor([i_sample]))
                    text_index = text_index.cuda() if use_cuda else text_index
                    textAttention = torch.index_select(myAttentions, dim=0, index=text_index)
                    layout_exp = expr_list[i_sample]

                    image_index = Variable(torch.LongTensor([i_sample]))
                    image_index = image_index.cuda() if use_cuda else image_index
                    ith_images_variable = torch.index_select(input_images_variable, dim=0, index=image_index)
                    # .view(1,D_feat,H_feat,W_feat)
                    myAnswers = myModuleNet(input_image_variable=ith_images_variable,
                                            input_text_attention_variable=textAttention,
                                            target_answer_variable=ith_answer_variable,
                                            expr_list=layout_exp)

                    current_answer = torch.topk(myAnswers, 1)[1].cpu().data.numpy()
                    if current_answer[0, 0] == input_answers[i_sample]:
                        n_correct_answer += 1


            if(print_log and i %20 ==0):
                curr_answer_accuracy = n_correct_answer/num_questions_total
                print(layout_correct_total ,num_questions_total, curr_answer_accuracy)
            # Assemble the layout tokens into network structure

            layout_valid_total += np.sum(expr_validity_array)

        answer_accuracy = n_correct_answer / num_questions_total
        layout_accuracy = layout_correct_total / num_questions_total
        layout_validity = layout_valid_total / num_questions_total




        return layout_accuracy, layout_correct_total ,num_questions_total, answer_accuracy