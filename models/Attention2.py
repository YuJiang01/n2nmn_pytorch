import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from global_variables.global_variables import use_cuda


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
                 max_decoder_len=0, dropout_p=0.1,num_layers = 1,
                 assembler_w=None, assembler_b=None, assembler_p = None,EOStoken=-1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_encoding_size = output_encoding_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.max_decoder_len = max_decoder_len

        self.go_embeding = nn.Embedding(1, self.output_encoding_size)
        self.embedding = nn.Embedding(self.output_size, self.output_encoding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.output_encoding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        self.encoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.attnLinear = nn.Linear(self.hidden_size, 1)
        self.assembler_w = torch.FloatTensor(assembler_w).cuda() if use_cuda else torch.FloatTensor(assembler_w)
        self.assembler_b = torch.FloatTensor(assembler_b).cuda() if use_cuda else torch.FloatTensor(assembler_b)
        self.assembler_p = torch.FloatTensor(assembler_p).cuda() if use_cuda else torch.FloatTensor(assembler_p)
        self.batch_size = 0
        self.EOS_token = EOStoken
        self._init_par()
    
    def _init_par(self):
        torch.nn.init.xavier_uniform(self.decoderLinear.weight)
        torch.nn.init.xavier_uniform(self.attnLinear.weight)
        torch.nn.init.constant(self.decoderLinear.bias,0)
        torch.nn.init.constant(self.attnLinear.bias,0)
    '''
        compute if a token is valid at current sequence
        decoding_state [N,3]
        assembler_w [3,output_size, 4 ]
        assembler_b [output_size, 4]
        output [N, output_size]
    '''
    def _get_valid_tokens(self,decoding_state, assembler_W, assembler_b):

        batch_size = decoding_state.size(0)
        expanded_state = decoding_state.view(batch_size,3,1,1).expand(batch_size, 3, self.output_size, 4)

        expanded_w= assembler_W.view(1,3, self.output_size,4).expand(batch_size, 3, self.output_size, 4)

        tmp1 = torch.sum(expanded_state * expanded_w, dim=1)
        expanded_b = assembler_b.view(1,-1,4).expand(batch_size,-1,4)
        tmp2= tmp1 - expanded_b
        tmp3 = torch.min(tmp2,dim=2)[0]
        token_invalidity = torch.lt(tmp3, 0)
        token_invalidity = token_invalidity.cuda() if use_cuda else token_invalidity
        return token_invalidity


    '''
        update the decoding state, which is used to determine if a token is valid
        decoding_state [N,3]
        assembler_p [output_size, 3]
        predicted_token [N,output_size]
        output [N, output_size]
    '''
    def _update_decoding_state(self, decoding_state, predicted_token, assembler_P):
        decoding_state = decoding_state + torch.mm(predicted_token , assembler_P)
        return decoding_state


    '''
        for a give state compute the lstm hidden layer, attention and predicted layers
        can handle the situation where seq_len is 1 or >1 (i.e., s=using groudtruth layout)
        
        input parameters :
            time: int, time step of decoder
            previous_token: [decoder_len, batch], decoder_len=1 for step-by-step decoder
            previous_hidden_state: (h_n, c_n), dimmension:both are (num_layers * num_directions, batch, hidden_size)
            encoder_outputs : outputs from LSTM in encoder[seq_len, batch, hidden_size * num_directions]
            encoder_lens: list of input sequence lengths
            decoding_state: the state used to decide valid tokens
        
        output parameters : 
            predicted_token: [decoder_len, batch]
            Att_weighted_text: batch,out_len,txt_embed_dim
            log_seq_prob: [batch]
            neg_entropy: [batch]
    '''
    def _step_by_step_attention_decoder(self, time, embedded, previous_hidden_state,
                                        encoder_outputs, encoder_lens, decoding_state,target_variable,sample_token):

        ##step1 run LSTM to get decoder hidden state
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

        attention = F.softmax(raw_attention, dim=2)  ##(batch,out_len,seq_len)
        

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



        ##get the valid token for current position based on previous token to perform a mask for next prediction
        ## token_validity [N, output_size]
        token_invalidity = self._get_valid_tokens(decoding_state=decoding_state,
                                                assembler_W=self.assembler_w,
                                                assembler_b=self.assembler_b)

        ## probs
        probs = output_prob.view(-1,self.output_size)
        probs.data.masked_fill_(token_invalidity,0.0)
        probs_sum = torch.sum(probs, dim=1, keepdim=True)
        probs = probs/probs_sum


        if target_variable is not None:
            predicted_token = target_variable[time, :].view(-1,1)
        elif sample_token:
            predicted_token = probs.multinomial()
        else:
            predicted_token = torch.max(probs, dim=1)[1].view(-1, 1)


        ##[batch_size, self.output_size]
        tmp = torch.zeros(batch_size, self.output_size)
        tmp = tmp.cuda() if use_cuda else tmp
        predicted_token_encoded = tmp.scatter_(1, predicted_token.data, 1.0)
        predicted_token_encoded = predicted_token_encoded.cuda() if use_cuda else predicted_token_encoded

        updated_decoding_state = self._update_decoding_state(decoding_state=decoding_state,
                                                             predicted_token=predicted_token_encoded,
                                                             assembler_P=self.assembler_p)

        ## compute the negative entropy
        token_invalidity_float = Variable(token_invalidity.type(torch.FloatTensor)).detach()
        token_invalidity_float = token_invalidity_float.cuda() if use_cuda else token_invalidity_float
        token_neg_entropy = torch.sum(probs.detach() * torch.log(probs + 0.000001), dim=1)

        ## compute log_seq_prob
        selected_token_log_prob =torch.log(torch.sum(probs * Variable(predicted_token_encoded), dim=1)+ 0.000001)


        return predicted_token.permute(1, 0), hidden, attention, updated_decoding_state,token_neg_entropy, selected_token_log_prob





    def forward(self,encoder_hidden,encoder_outputs,encoder_lens,target_variable,sample_token):
        self.batch_size = encoder_outputs.size(1)
        total_neg_entropy = 0
        total_seq_prob = 0

        ## set initiate step:
        time = 0
        start_token = Variable(torch.LongTensor(np.zeros((1, self.batch_size))), requires_grad=False)
        start_token = start_token.cuda() if use_cuda else start_token
        next_input = self.go_embeding(start_token)
        next_decoding_state = torch.FloatTensor([[0, 0, self.max_decoder_len]]).expand(self.batch_size, 3).contiguous()
        next_decoding_state = next_decoding_state.cuda() if use_cuda else next_decoding_state
        loop_state = True
        previous_hidden = encoder_hidden

        while time < self.max_decoder_len :
            predicted_token, previous_hidden, context, next_decoding_state, neg_entropy, log_seq_prob = \
                self._step_by_step_attention_decoder(time=time,
                    embedded= next_input,
                    previous_hidden_state=previous_hidden, encoder_outputs=encoder_outputs,
                    encoder_lens=encoder_lens, decoding_state=next_decoding_state,target_variable= target_variable,sample_token=sample_token)

            if time == 0:
                predicted_tokens = predicted_token
                total_neg_entropy = neg_entropy
                total_seq_prob = log_seq_prob
                context_total = context
            else:
                predicted_tokens = torch.cat((predicted_tokens, predicted_token))
                total_neg_entropy += neg_entropy
                total_seq_prob += log_seq_prob
                context_total = torch.cat((context_total, context), dim=1)

            time +=1
            next_input =self.embedding(predicted_token)
            loop_state = torch.ne(predicted_token, self.EOS_token).any()

        return predicted_tokens, context_total, total_neg_entropy, total_seq_prob




class attention_seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(attention_seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs,input_seq_lens,target_variable,sample_token):
        encoder_hidden = self.encoder.initHidden(len(input_seq_lens))
        encoder_outputs, encoder_hidden, txt_embedded = self.encoder(input_seqs,input_seq_lens, encoder_hidden)
        decoder_results, attention, neg_entropy, log_seq_prob = self.decoder(target_variable=target_variable,
                                                                    encoder_hidden= encoder_hidden,
                                                                    encoder_outputs= encoder_outputs,
                                                                    encoder_lens=input_seq_lens, sample_token=sample_token
                                                                             )
        ##using attention from decoder and txt_embedded from the encoder to get the attention weighted text
        ## txt_embedded [seq_len,batch,input_encoding_size]
        ## attention [batch, out_len,seq_len]
        txt_embedded_perm = txt_embedded.permute(1,0,2)
        att_weighted_text = torch.bmm(attention, txt_embedded_perm)
        
        
        return decoder_results, att_weighted_text, neg_entropy, log_seq_prob
        #return decoder_results, attention, neg_entropy, log_seq_prob
