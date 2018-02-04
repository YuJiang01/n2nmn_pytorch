import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import random

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 500

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input_seqs, input_seq_lens, hidden):
        embedded = self.embedding(input_seqs)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_seq_lens)
        outputs, hidden = self.lstm(embedded)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,num_layers = 1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, target_variable, hidden, encoder_outputs):
        embedded = self.embedding(target_variable)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        ##step1 run LSTM to get decoder hidden state
        output, hidden = self.lstm(embedded, hidden)

        ##step2: use function in Eq(2) of the paper to compute attention
        ##(seq_len,batch,hidden_size) -->(batch,hidden_size,seq_len)
        encoder_outputs = encoder_outputs.permute(1,2,0)

        ##(out_len,batch,hidden_size) --> (batch,out_len,hidden_size)
        output = output.permute(1,0,2)

        ## (batch,out_len,hidden_size) * (batch,hidden_size,seq_len) => (batch,out_len,seq_len)
        attention = torch.bmm(output,encoder_outputs)
        attention = F.softmax(attention.view(-1,seq_len)).view(batch_size,-1,seq_len)

        ##(batch,hidden_size,seq_len)-->(batch,seq_len,hidden_size)
        encoder_outputs = encoder_outputs.permute(0,2,1)

        ##(batch,out_len,seq_len) * (batch,seq_len,hidden_size) --> (batch,out_len,hidden_size)
        mix = torch.bmm(attention,encoder_outputs)


        ##(batch,out_len,hidden_size*2)
        combined = torch.cat((mix,output),dim=2)

        ##(batch,out_len,hidden_size)
        output = F.tanh(self.attn_combine(combined.view(-1,2*hidden_size))).view(batch_size,-1,hidden_size)

        ##(batch,out_len,hidden_size) ->(out_len,batch,hidden_size)
        output = output.permute(1,0,2)

        ##
        output = F.softmax(self.out(output),dim = 2)
        return output, hidden, attention

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result





class attention_seq2seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(attention_seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs,input_seq_lens,target_variable):
        encoder_hidden = self.encoder.initHidden(len(input_seq_lens))
        encoder_outputs,encoder_hidden = self.encoder(input_seqs,input_seq_lens,encoder_hidden)
        decoder_results,encoder_hidden,attention = self.decoder(target_variable,encoder_hidden,encoder_outputs)
        return decoder_results,attention











