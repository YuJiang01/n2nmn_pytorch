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

def train_layout(input_text_seq,input_layout,encoder,decoder,criterion):
    #input_variable = Variable(torch.LongTensor(input_text_seq))
    #target_variable = Variable(torch.LongTensor(input_layout))

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


    loss, decoder_out, decoder_attention = train(input_text_seq, input_layout, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion)

    return loss, decoder_out, decoder_attention



def run_training(max_iter, dataset_trn):
    avg_accuracy = 0
    accuracy_decay = 0.99

    ##get the dimension of answer

    num_vocab_txt = dataset_trn.batch_loader.vocab_dict.num_vocab
    num_vocab_nmn = len(assembler.module_names)
    num_choices = dataset_trn.batch_loader.answer_dict.num_vocab
    myEncoder = EncoderRNN(num_vocab_txt, hidden_size)
    myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn)
    criterion = nn.NLLLoss()

    if use_cuda:
        myEncoder = myEncoder.cuda()
        myDecoder = myDecoder.cuda()

    n_correct_total = 0
    n_total = 0
    for n_iter, batch in enumerate(dataset_trn.batches()):
        if n_iter >= max_iter:
            break

        ##consider current model, run sample one by one

        _, n_sample = batch['input_seq_batch'].shape

        n_total += n_sample

        n_correct_layout = 0
        for i_sample in range(n_sample):

            input_text_seq_len = batch['seq_length_batch'][i_sample]
            input_text_seq = batch['input_seq_batch'][:input_text_seq_len, i_sample]
            input_image_feat = batch['image_feat_batch'][i_sample,:,:,:]
            input_layout = batch['gt_layout_batch'][:,i_sample]

            loss, decoder_out, decoder_attention= train_layout(input_text_seq,input_layout,myEncoder,myDecoder,criterion)
            if (np.array(decoder_out) == input_layout).all():
                n_correct_layout += 1

        mini_batch_accuracy = n_correct_layout/n_sample
        n_correct_total += n_correct_layout
        avg_accuracy = n_correct_total/n_total

        if n_iter % 50 ==0:
            print("iter:",n_iter, " cur_accuracy:" ,mini_batch_accuracy," avg_accuracy:",avg_accuracy)
            sys.stdout.flush()


        # Save snapshot
        if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
            snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
            #snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
            print('snapshot saved to ' + snapshot_file)
