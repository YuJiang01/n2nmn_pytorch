from __future__ import absolute_import, division, print_function

from Utils.data_reader import DataReader

import sys
from models.layout_assembler import Assembler
from train_model.input_parameters import *
import torch
import numpy as np
from models.module_net import  module_net

use_cuda = torch.cuda.is_available()

from models.AttensionSeq2Seq import *



##create directory for snapshot
os.makedirs(snapshot_dir, exist_ok=True)


assembler = Assembler(vocab_layout_file)


data_reader_trn = DataReader(imdb_file_trn, shuffle=True, one_pass=False,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file,
                             prune_filter_module=prune_filter_module)

num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab





### running the step for attenstionSeq2Seq

avg_accuracy = 0
accuracy_decay = 0.99


num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab

myEncoder = EncoderRNN(num_vocab_txt, hidden_size, num_layers)
myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn, 0.1, num_layers)

criterion = nn.NLLLoss()
criterion_answer = nn.NLLLoss()

if use_cuda:
    myEncoder = myEncoder.cuda()
    myDecoder = myDecoder.cuda()

mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq


myModuleNet = module_net(image_height= H_feat, image_width = W_feat, in_image_dim=D_feat,
                         in_text_dim=embed_dim_txt, out_num_choices= num_choices, map_dim=hidden_size)

answerOptimizer = optim.Adam(myModuleNet.parameters())
answerOptimizer.zero_grad()

n_correct_total = 0
n_total = 0

for i_iter, batch in enumerate(data_reader_trn.batches()):
    if i_iter >= max_iter:
        break

    ##consider current model, run sample one by one

    _, n_sample = batch['input_seq_batch'].shape
    input_text_seq_lens = batch['seq_length_batch']
    input_text_seqs = batch['input_seq_batch']
    input_layouts = batch['gt_layout_batch']
    input_images = batch['image_feat_batch']
    input_answers = batch['answer_label_batch']

    n_total += n_sample

    n_correct_layout = 0

    input_variable = Variable(torch.LongTensor(input_text_seqs))
    input_variable = input_variable.cuda() if use_cuda else input_variable

    target_variable = Variable(torch.LongTensor(input_layouts))
    target_variable = target_variable.cuda() if use_cuda else target_variable

    myOptimizer = optim.Adam(mySeq2seq.parameters())
    myOptimizer.zero_grad()

    myLayouts, myAttentions = mySeq2seq(input_variable, input_text_seq_lens, target_variable)

    # Get loss
    layout_loss = 0
    for step, step_output in enumerate(myLayouts):
        layout_loss += criterion(step_output.view(n_sample, -1), target_variable[step, :])

    layout_loss.backward()
    myOptimizer.step()

    ##compute accuracy
    predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]
    n_correct_layout += np.sum(np.all(predicted_layouts == input_layouts,axis=0))


    ## get the layout information
    expr_list, expr_validity_array = assembler.assemble(predicted_layouts)

    input_answers_variable = Variable(torch.LongTensor(input_answers))
    input_answers_variable = input_answers_variable.cuda() if use_cuda else input_answers_variable

    input_images_variable = Variable(torch.LongTensor(input_images))
    input_images_variable = input_images_variable.cuda() if use_cuda else input_images_variable
    ## for the first step, train sample one by one
    for i_sample in range(n_sample):
        ##skip the case when expr is not validity
        if expr_validity_array[i_sample]:
            ith_answer_variable = input_answers_variable[i_sample]
            textAttention = myAttentions
            layout_exp = expr_list[i_sample]
            ith_images_variable = input_images_variable[i_sample, :, :, :]
            myAnswers = myModuleNet(input_image_variable = ith_images_variable,
                                    input_text_attention_variable = textAttention,
                                    target_answer_variable = ith_answer_variable,
                                    expr_list = layout_exp,expr_validity=expr_validity_array[i_sample])




    #myAnswers = myModuleNet(input_images_variable, myAttentions,input_answers_variable,expr_list,expr_validity_array)

    #answer_loss = criterion_answer(myAnswers, input_answers_variable)

    #answer_loss.backward()
    #answerOptimizer.step()

    n_correct_layout += np.sum(np.all(predicted_layouts == input_layouts, axis=0))
    mini_batch_accuracy = n_correct_layout / n_sample
    n_correct_total += n_correct_layout
    avg_accuracy = n_correct_total / n_total

    if (i_iter + 1) % log_interval == 0 or i_iter < 50 :
        print("iter:", i_iter, " cur_accuracy:", mini_batch_accuracy, " avg_accuracy:", avg_accuracy)
        sys.stdout.flush()

    # Save snapshot
    if  (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (i_iter + 1))
        torch.save(mySeq2seq, snapshot_file)
        print('snapshot saved to ' + snapshot_file)







