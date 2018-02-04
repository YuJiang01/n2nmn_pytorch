import numpy as np
import sys
from models.layout_assembler import Assembler
from train_model.input_parameters import *
import torch

use_cuda = torch.cuda.is_available()

from models.AttensionSeq2Seq import *

snapshot_dir = './exp_clevr/tfmodel/%s/' % exp_name

assembler = Assembler(vocab_layout_file)

hidden_size = 512
learning_rate = 0.01



def run_training(max_iter, dataset_trn):
    avg_accuracy = 0
    accuracy_decay = 0.99

    ##get the dimension of answer

    num_vocab_txt = dataset_trn.batch_loader.vocab_dict.num_vocab
    num_vocab_nmn = len(assembler.module_names)
    num_choices = dataset_trn.batch_loader.answer_dict.num_vocab
    myEncoder = EncoderRNN(num_vocab_txt, hidden_size,num_layers)
    myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn,0.1,num_layers)

    criterion = nn.NLLLoss()

    if use_cuda:
        myEncoder = myEncoder.cuda()
        myDecoder = myDecoder.cuda()

    mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
    mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq

    n_correct_total = 0
    n_total = 0
    loss = 0

    for n_iter, batch in enumerate(dataset_trn.batches()):
        if n_iter >= max_iter:
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
        myLayouts,myAttentions = mySeq2seq(input_variable,input_text_seq_lens,target_variable)

        # Get loss
        loss = 0
        for step, step_output in enumerate(myLayouts):
            loss += criterion(step_output.view(n_sample, -1), target_variable[step ,:])

        loss.backward()
        myOptimizer.step()

        ##compute accuracy
        predicted_layouts = torch.topk(myLayouts, 1)[1].data.numpy()
        for i in range(n_sample):
            x1 = predicted_layouts[:,i,0].tolist()
            x2 = input_layouts[:,i].tolist()
            if x1==x2:
                n_correct_layout += 1

        mini_batch_accuracy = n_correct_layout/n_sample
        n_correct_total += n_correct_layout
        avg_accuracy = n_correct_total/n_total

        if n_iter % 1 ==50:
            print("iter:",n_iter, " cur_accuracy:" ,mini_batch_accuracy," avg_accuracy:",avg_accuracy)
            sys.stdout.flush()


        # Save snapshot
        if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
            snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
            
            print('snapshot saved to ' + snapshot_file)
