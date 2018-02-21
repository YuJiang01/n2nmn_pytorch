import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import random
import numpy as np

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 500


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, input_encoding_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, input_encoding_size)
        self.lstm = nn.LSTM(input_encoding_size, hidden_size)

    def forward(self, input_seqs, input_seq_lens, hidden):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden, embedded

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,output_encoding_size, dropout_p=0.1,num_layers = 1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_encoding_size = output_encoding_size
        self.dropout_p = dropout_p
        self.max_length = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.output_encoding_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.output_encoding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        self.encoderLinear = nn.Linear(self.hidden_size,self.hidden_size)
        self.decoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.attnLinear = nn.Linear(self.hidden_size, 1)


    def forward(self, target_variable, hidden, encoder_outputs,encoder_lens = None):
        embedded = self.embedding(target_variable)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)
        out_len = embedded.size(0)

        ##step1 run LSTM to get decoder hidden state
        output, hidden = self.lstm(embedded, hidden)

        ##step2: use function in Eq(2) of the paper to compute attention
        ##size encoder_outputs (seq_len,batch_size,hidden_size)==>(out_len,seq_len,batch_size,hidden_size)
        encoder_outputs_expand = encoder_outputs.view(1,seq_len,batch_size,hidden_size).expand(out_len,seq_len,batch_size,hidden_size)
        encoder_transform = self.encoderLinear(encoder_outputs_expand)

        ##size output (out_len,batch_size,hidden_size)
        output_expand =output.view(out_len,1,batch_size,hidden_size).expand(out_len,seq_len,batch_size,hidden_size)
        output_transfrom = self.decoderLinear(output_expand)

        ##raw_attention size (out_len,seq_len,batch_size,1)
        raw_attention = self.attnLinear(F.tanh(encoder_transform + output_transfrom)).view(out_len,seq_len,batch_size)  ## Eq2

        #(out_len, seq_len, batch_size)==>(batch_size,out_len,seq_len)
        raw_attention = raw_attention.permute(2, 0, 1)

        ##mask the end of the question
        if encoder_lens is not None:
            mask = np.ones((batch_size, out_len, seq_len))
            for i, v in enumerate(encoder_lens):
                mask[i, :, 0:v] = 0
            mask_tensor = torch.ByteTensor(mask)
            mask_tensor = mask_tensor.cuda() if use_cuda else mask_tensor
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))

        attention = F.softmax(raw_attention, dim=2) ##(batch,out_len,seq_len)  TODO: double check which dim to do softmax


        ##c_t = \sum_{i=1}^I att_{ti}h_i t: decoder time t, and encoder time i
        ## (seq_len,batch_size,hidden_size) ==>(batch_size,seq_len,hidden_size)
        encoder_batch_first = encoder_outputs.permute(1,0,2)
        context = torch.bmm(attention, encoder_batch_first)

        ##(out_len,batch,hidden_size) --> (batch,out_len,hidden_size)
        output_batch_first = output.permute(1, 0, 2)

        ##(batch,out_len,hidden_size*2)
        combined = torch.cat((context, output_batch_first), dim=2).permute(1, 0, 2) ##TODO continue here

        output_prob = F.softmax(self.out(combined), dim=2)

        ##(batch,out_len,hidden_size) ->(out_len,batch,hidden_size)
        #output_prob = output_prob.permute(1, 0, 2)
        return output_prob, hidden, context

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result





class attention_seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(attention_seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs,input_seq_lens,target_variable):
        encoder_hidden = self.encoder.initHidden(len(input_seq_lens))
        encoder_outputs, encoder_hidden, txt_embeded = self.encoder(input_seqs,input_seq_lens, encoder_hidden)
        decoder_results, encoder_hidden, attention = self.decoder(target_variable, encoder_hidden,encoder_outputs,input_seq_lens)

        ##(seq_len,batch,txt_embed_dim) ==> (batch,seq_len,txt_embed_dim)
        #txt_embeded_perm = txt_embeded.permute(1,0,2)
        ##(batch,out_len,seq_len) * (batch,seq_len,txt_embed_dim) --> (batch,out_len,txt_embed_dim)
        #att_weigted_text = torch.bmm(attention, txt_embeded_perm)

        #return decoder_results,att_weigted_text
        return decoder_results, attention











