import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import random
import numpy as np


use_cuda = torch.cuda.is_available()
SOS_token = 1
EOS_token = 0

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
    def __init__(self, hidden_size, output_size, output_encoding_size,
                 max_decoder_len=10, dropout_p=0.1,num_layers = 1,
                 assembler_w=None, assembler_b = None, assembler_p = None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_encoding_size = output_encoding_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.max_decoder_len = max_decoder_len

        self.embedding = nn.Embedding(self.output_size, self.output_encoding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.output_encoding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        self.encoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.attnLinear = nn.Linear(self.hidden_size, 1)
        self.assembler_w = torch.FloatTensor(assembler_w)
        self.assembler_b = torch.FloatTensor(assembler_b)
        self.assembler_p = torch.FloatTensor(assembler_p)


    '''decoding_state [N,3]
    assembler_w [3,output_size, 4 ]
    assembler_b [output_size, 4]
    output [N, output_size]
    '''
    def _get_valid_tokens(self,decoding_state, assembler_W, assembler_b):

        batch_size = decoding_state.size(0)
        expanded_state=decoding_state.view(-1,3,1,1).expand(-1, 3, self.output_size, 4)

        expanded_w= assembler_W.view(1,3, self.output_size,4).expand(batch_size, 3, self.output_size, 4)

        tmp1 = torch.sum(expanded_state * expanded_w, dim=1)
        expanded_b = assembler_b.view(1,-1,4).expand(batch_size,-1,4)
        tmp2= tmp1 - expanded_b
        tmp3 = torch.min(tmp2,dim=2)[0]
        token_validity = torch.gt(tmp3, 0)

        return token_validity


    '''decoding_state [N,3]
    assembler_p [output_size, 3]
    predicted_token [N,output_size]
    output [N, output_size]
    '''
    def _update_decoding_state(self, decoding_state, predicted_token, assembler_P):
        decoding_state = decoding_state + torch.mm(predicted_token , assembler_P)
        return decoding_state

    ## this function handle the case decode each step or a whole sequence
    def _step_by_step_attention_decoder(self, time, previous_token, previous_hidden_state, encoder_outputs,decoding_state=None,encoder_lens = None):
        ##step1 run LSTM to get decoder hidden state
        embedded = self.embedding(previous_token)
        embedded = self.dropout(embedded)

        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)
        out_len = embedded.size(0)

        output, hidden = self.lstm(embedded, previous_hidden_state)
        ##step2: use function in Eq(2) of the paper to compute attention
        ##size encoder_outputs (seq_len,batch_size,hidden_size)==>(out_len,seq_len,batch_size,hidden_size)
        encoder_outputs_expand = encoder_outputs.view(1, seq_len, batch_size, hidden_size).expand(out_len, seq_len,
                                                                                                  batch_size,
                                                                                                  hidden_size)
        encoder_transform = self.encoderLinear(encoder_outputs_expand)

        ##size output (out_len,batch_size,hidden_size)
        output_expand = output.view(out_len, 1, batch_size, hidden_size).expand(out_len, seq_len, batch_size,
                                                                                hidden_size)
        output_transfrom = self.decoderLinear(output_expand)

        ##raw_attention size (out_len,seq_len,batch_size,1)
        raw_attention = self.attnLinear(F.tanh(encoder_transform + output_transfrom)).view(out_len, seq_len,
                                                                                           batch_size)  ## Eq2

        # (out_len, seq_len, batch_size)==>(batch_size,out_len,seq_len)
        raw_attention = raw_attention.permute(2, 0, 1)

        ##mask the end of the question
        if encoder_lens is not None:
            mask = np.ones((batch_size, out_len, seq_len))
            for i, v in enumerate(encoder_lens):
                mask[i, :, 0:v] = 0
            mask_tensor = torch.ByteTensor(mask)
            mask_tensor = mask_tensor.cuda() if use_cuda else mask_tensor
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))

        attention = F.softmax(raw_attention,
                              dim=2)  ##(batch,out_len,seq_len)  TODO: double check which dim to do softmax

        ##c_t = \sum_{i=1}^I att_{ti}h_i t: decoder time t, and encoder time i
        ## (seq_len,batch_size,hidden_size) ==>(batch_size,seq_len,hidden_size)
        encoder_batch_first = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attention, encoder_batch_first)

        ##(out_len,batch,hidden_size) --> (batch,out_len,hidden_size)
        output_batch_first = output.permute(1, 0, 2)

        ##(batch,out_len,hidden_size*2)
        combined = torch.cat((context, output_batch_first), dim=2).permute(1, 0, 2)

        ## [out_len,batch,out_size]
        output_prob = F.softmax(self.out(combined), dim=2)

        ##get the valid token for current position based on previous token
        ## token_validity [N, output_size]
        token_validity = self._get_valid_tokens(decoding_state=decoding_state,
                                              assembler_W=self.assembler_w,
                                              assembler_b=self.assembler_b)

        token_validity = Variable(token_validity.type(torch.FloatTensor).view(-1,batch_size ,self.output_size)).detach()

        probs = F.softmax(token_validity * output_prob, dim=2).view(-1,self.output_size)

        ## predict the token for current layer,
        predicted_token = probs.multinomial()


        ##[batch_size, self.output_size]
        predicted_token_encoded = torch.zeros(batch_size, self.output_size).scatter_(1, predicted_token.data, 1)

        updated_decoding_state = self._update_decoding_state(decoding_state=decoding_state,
                                                             predicted_token=predicted_token_encoded,
                                                             assembler_P=self.assembler_p)

        return predicted_token.permute(1,0), hidden, output_prob, context,updated_decoding_state


    def forward(self, hidden, encoder_outputs, encoder_lens, target_variable = None ):

        batch_size = encoder_outputs.size(1)

        if target_variable is None:
            output_prob_total = []
            context_total = []
            time = 0;
            loop_state = True
            previous_token = Variable(torch.LongTensor(np.zeros((1,batch_size))))
            previous_hidden = hidden
            next_decoding_state = torch.FloatTensor(batch_size, 3).zero_()

            while time < self.max_decoder_len and loop_state :
                predicted_token, previous_hidden, output_prob, context, next_decoding_state = \
                    self._step_by_step_attention_decoder(
                    time=time,previous_token=previous_token,
                    previous_hidden_state=previous_hidden, encoder_outputs=encoder_outputs,
                    encoder_lens=encoder_lens,decoding_state=next_decoding_state)
                time +=1
                previous_token = predicted_token
                loop_state = torch.ne(predicted_token, EOS_token).any()
                output_prob_total.append(output_prob)
                context_total.append(context)


            return output_prob_total, previous_hidden, context_total


        else:
            predicted_token, hidden, output_prob, context,_ = \
                self._step_by_step_attention_decoder(
                    time=0, previous_token=target_variable,
                    previous_hidden_state=hidden, encoder_outputs=encoder_outputs,
                    encoder_lens=encoder_lens)
            return output_prob, hidden, context




    '''
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
        return output_prob, hidden, context'''

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
        decoder_results, encoder_hidden, attention = self.decoder(target_variable=None,
                                                                    hidden= encoder_hidden,
                                                                    encoder_outputs= encoder_outputs,
                                                                    encoder_lens=input_seq_lens)

        ##(seq_len,batch,txt_embed_dim) ==> (batch,seq_len,txt_embed_dim)
        #txt_embeded_perm = txt_embeded.permute(1,0,2)
        ##(batch,out_len,seq_len) * (batch,seq_len,txt_embed_dim) --> (batch,out_len,txt_embed_dim)
        #att_weigted_text = torch.bmm(attention, txt_embeded_perm)

        #return decoder_results,att_weigted_text
        return decoder_results, attention











