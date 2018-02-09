from __future__ import absolute_import, division, print_function

from Utils.data_reader import DataReader

import sys
from models.layout_assembler import Assembler
from train_model.input_parameters import *
import torch
import numpy as np
from models.module_net import module_net
from Utils.utils import unique_columns

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



num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab

myEncoder = EncoderRNN(num_vocab_txt, hidden_size, num_layers)
myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn, 0.1, num_layers)

criterion = nn.NLLLoss()
criterion_answer = nn.CrossEntropyLoss()

if use_cuda:
    myEncoder = myEncoder.cuda()
    myDecoder = myDecoder.cuda()

mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq


myModuleNet = module_net(image_height= H_feat, image_width = W_feat, in_image_dim=D_feat,
                         in_text_dim=embed_dim_txt, out_num_choices= num_choices, map_dim=hidden_size)

myModuleNet = myModuleNet.cuda() if use_cuda else myModuleNet

answerOptimizer = optim.Adam(myModuleNet.parameters())


avg_accuracy = 0
accuracy_decay = 0.99

n_correct_layout_total = 0
n_total = 0
n_correct_answer_total = 0

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
    n_correct_answer = 0

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

    #layout_loss.backward(retain_graph=True)
    #myOptimizer.step()

    ##compute accuracy
    predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]
    n_correct_layout += np.sum(np.all(predicted_layouts == input_layouts,axis=0))


    ## get the layout information
    expr_list, expr_validity_array = assembler.assemble(predicted_layouts)

    #input_answers_variable = Variable(torch.LongTensor(input_answers))
    #input_answers_variable = input_answers_variable.cuda() if use_cuda else input_answers_variable

    #input_images_variable = Variable(torch.FloatTensor(input_images))
    #input_images_variable = input_images_variable.cuda() if use_cuda else input_images_variable

    ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
    #input_images_variable = input_images_variable.permute(0,3,1,2)



    answer_loss = 0
    ##select the samples with the same predicted layouts and train together

    sample_groups_by_layout = unique_columns(predicted_layouts)

    for sample_group in sample_groups_by_layout:
        if sample_group.shape ==0: continue

        answerOptimizer.zero_grad()
        first_in_group = sample_group[0]
        if expr_validity_array[first_in_group]:
            layout_exp = expr_list[first_in_group]
            ith_answer = input_answers[sample_group]

            ith_answer_variable = Variable(torch.LongTensor(ith_answer))
            ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable


            textAttention = myAttentions[sample_group, :]

            ith_image = input_images[sample_group,:,:,:]
            ith_images_variable = Variable(torch.FloatTensor(ith_image))
            ith_images_variable = ith_images_variable.cuda() if use_cuda else ith_images_variable

            ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
            ith_images_variable = ith_images_variable.permute(0, 3, 1, 2)

            ith_images_variable = ith_images_variable.contiguous()

            myAnswers = myModuleNet(input_image_variable=ith_images_variable,
                                    input_text_attention_variable=textAttention,
                                    target_answer_variable=ith_answer_variable,
                                    expr_list=layout_exp)
            answer_loss += criterion_answer(myAnswers, ith_answer_variable)

            current_answer = torch.topk(myAnswers, 1)[1].cpu().data.numpy()[:,0]

            n_correct_answer += np.sum(current_answer == input_answers[sample_group])



    ## for the first step, train sample one by one
    '''for i_sample in range(n_sample):

        ##skip the case when expr is not validity
        if expr_validity_array[i_sample]:
            answerOptimizer.zero_grad()
            #print("iter:", i_iter, " isample:" ,i_sample)
            ith_answer_variable = input_answers_variable[i_sample]
            ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable
            text_index = Variable(torch.LongTensor([i_sample]))
            text_index = text_index.cuda() if use_cuda else text_index
            textAttention = torch.index_select(myAttentions, dim=0,index=text_index)
            layout_exp = expr_list[i_sample]

            image_index = Variable(torch.LongTensor([i_sample]))
            image_index = image_index.cuda() if use_cuda else image_index
            ith_images_variable = torch.index_select(input_images_variable, dim=0,index=image_index)
            #.view(1,D_feat,H_feat,W_feat)
            myAnswers = myModuleNet(input_image_variable = ith_images_variable,
                                    input_text_attention_variable =textAttention,
                                    target_answer_variable = ith_answer_variable,
                                    expr_list=layout_exp,expr_validity=expr_validity_array[i_sample])

            answer_loss += criterion_answer(myAnswers,ith_answer_variable)
            current_answer = torch.topk(myAnswers, 1)[1].cpu().data.numpy()
            if current_answer[0, 0] == input_answers[i_sample]:
                n_correct_answer += 1
            #answer_loss.backward(retain_graph=True)
            #answerOptimizer.step()'''

    total_loss = layout_loss+ answer_loss
    total_loss.backward()
    myOptimizer.step()
    answerOptimizer.step()





    current_layout_accuracy = n_correct_layout/n_sample
    n_correct_layout_total += n_correct_layout
    avg_layout_accuracy = n_correct_layout_total / n_total

    current_answer_accuracy = n_correct_answer/n_sample
    n_correct_answer_total += n_correct_answer
    avg_answer_accuracy =  n_correct_answer_total/n_total

    if (i_iter + 1) % log_interval == 0 :
        print("iter:", i_iter + 1,
              " cur_layout_accuracy:%.3f"% current_layout_accuracy, " avg_layout_accuracy:%.3f"% avg_layout_accuracy,
              " cur_ans_accuracy:%.4f"% current_answer_accuracy, " avg_answer_accuracy:%.4f"% avg_answer_accuracy)
              #" layout_loss:%.3f"% layout_loss.cpu().data.numpy()[0], "answer_loss: %.3f"% answer_loss.cpu().data.numpy()[0])
        sys.stdout.flush()

    # Save snapshot
    if  (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
        layout_snapshot_file = os.path.join(snapshot_dir, "seq2seq_%08d" % (i_iter + 1))
        torch.save(mySeq2seq, layout_snapshot_file)
        module_snapshot_file = os.path.join(snapshot_dir, "module_%08d" % (i_iter + 1))
        torch.save(myModuleNet, module_snapshot_file)
        print('snapshot saved to ' + layout_snapshot_file + "and " +module_snapshot_file)







