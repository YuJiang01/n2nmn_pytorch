import torch
import torch.nn as nn
from models.AttensionSeq2Seq import *
from models.module_net import *
from Utils.utils import unique_columns


class end2endModuleNet(nn.Module):
    def __init__(self, num_vocab_txt, num_vocab_nmn, out_num_choices,
                embed_dim_nmn, embed_dim_txt, image_height, image_width, in_image_dim,
                hidden_size, assembler, layout_criterion, answer_criterion, num_layers=1, decoder_dropout=0.1):

        super(end2endModuleNet, self).__init__()

        self.assembler = assembler
        self.layout_criterion = layout_criterion
        self.answer_criterion = answer_criterion

        ##initiate encoder and decoder
        myEncoder = EncoderRNN(num_vocab_txt, hidden_size, embed_dim_txt, num_layers)
        myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn, embed_dim_nmn, decoder_dropout, num_layers)

        if use_cuda:
            myEncoder = myEncoder.cuda()
            myDecoder = myDecoder.cuda()


        ##initatiate attentionSeq2seq
        mySeq2seq = attention_seq2seq(myEncoder, myDecoder)
        self.mySeq2seq = mySeq2seq.cuda() if use_cuda else mySeq2seq


        ##initiate moduleNet
        myModuleNet = module_net(image_height=image_height, image_width=image_width, in_image_dim=in_image_dim,
                                 in_text_dim=embed_dim_txt, out_num_choices=out_num_choices, map_dim=hidden_size)

        self.myModuleNet = myModuleNet.cuda() if use_cuda else myModuleNet



    def forward(self, input_txt_variable, input_text_seq_lens, input_layout_variable, input_answers, input_images):

        batch_szie = len(input_text_seq_lens)

        ##run attentionSeq2Seq
        myLayouts, myAttentions = self.mySeq2seq(input_txt_variable, input_text_seq_lens, input_layout_variable)

        layout_loss = 0
        for step, step_output in enumerate(myLayouts):
            layout_loss +=self.layout_criterion(step_output.view(batch_szie, -1), input_layout_variable[step, :])


        predicted_layouts = torch.topk(myLayouts, 1)[1].cpu().data.numpy()[:, :, 0]
        expr_list, expr_validity_array = self.assembler.assemble(predicted_layouts)

        ## group samples based on layout
        sample_groups_by_layout = unique_columns(predicted_layouts)

        ##run moduleNet
        answer_loss = 0
        current_answer = np.zeros(batch_szie)

        for sample_group in sample_groups_by_layout:
            if sample_group.shape == 0:
                continue

            first_in_group = sample_group[0]
            if expr_validity_array[first_in_group]:
                layout_exp = expr_list[first_in_group]
                ith_answer = input_answers[sample_group]

                ith_answer_variable = Variable(torch.LongTensor(ith_answer))
                ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable

                textAttention = myAttentions[sample_group, :]

                ith_image = input_images[sample_group, :, :, :]
                ith_images_variable = Variable(torch.FloatTensor(ith_image))
                ith_images_variable = ith_images_variable.cuda() if use_cuda else ith_images_variable

                ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
                ith_images_variable = ith_images_variable.permute(0, 3, 1, 2)

                ith_images_variable = ith_images_variable.contiguous()

                myAnswers = self.myModuleNet(input_image_variable=ith_images_variable,
                                        input_text_attention_variable=textAttention,
                                        target_answer_variable=ith_answer_variable,
                                        expr_list=layout_exp)

                answer_loss += self.answer_criterion(myAnswers, ith_answer_variable)

                current_answer[sample_group] = torch.topk(myAnswers, 1)[1].cpu().data.numpy()[:, 0]

        #total_loss = layout_loss + answer_loss


        return layout_loss, answer_loss, current_answer, predicted_layouts, expr_validity_array






