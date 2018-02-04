from __future__ import absolute_import, division, print_function

from Utils.data_reader import DataReader

import sys
from models.layout_assembler import Assembler
from train_model.input_parameters import *
import torch
import numpy as np

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


myEncoder = EncoderRNN(num_vocab_txt,hidden_size)
myDecoder = AttnDecoderRNN(hidden_size,num_vocab_nmn)

avg_accuracy = 0
accuracy_decay = 0.99


num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab

myEncoder = EncoderRNN(num_vocab_txt, hidden_size, num_layers)
myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn, 0.1, num_layers)

criterion = nn.NLLLoss()

if use_cuda:
    myEncoder = myEncoder.cuda()
    myDecoder = myDecoder.cuda()

mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq

n_correct_total = 0
n_total = 0
loss = 0

for i_iter, batch in enumerate(data_reader_trn.batches()):
    if i_iter >= max_iter:
        break

    ##consider current model, run sample one by one

    _, n_sample = batch['input_seq_batch'].shape
    input_text_seq_lens = batch['seq_length_batch']
    input_text_seqs = batch['input_seq_batch']
    input_layouts = batch['gt_layout_batch']

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
    loss = 0
    for step, step_output in enumerate(myLayouts):
        loss += criterion(step_output.view(n_sample, -1), target_variable[step, :])

    loss.backward()
    myOptimizer.step()

    ##compute accuracy
    predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]
    n_correct_layout += np.sum(np.all(predicted_layouts == input_layouts,axis=0))

    mini_batch_accuracy = n_correct_layout / n_sample
    n_correct_total += n_correct_layout
    avg_accuracy = n_correct_total / n_total

    if (i_iter + 1) % log_interval == 0 or i_iter < 10:
        print("iter:", i_iter, " cur_accuracy:", mini_batch_accuracy, " avg_accuracy:", avg_accuracy)
        sys.stdout.flush()

    # Save snapshot
    if (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (i_iter + 1))
        torch.save(mySeq2seq, snapshot_file)
        print('snapshot saved to ' + snapshot_file)







