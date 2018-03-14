import torch
import torch.nn as nn
import sys
from models.Attention2 import *
from models.module_net import *
from Utils.utils import unique_columns




class end2endModuleNet(nn.Module):
    def __init__(self, num_vocab_txt, num_vocab_nmn, out_num_choices,
                embed_dim_nmn, embed_dim_txt, image_height, image_width, in_image_dim,
                hidden_size, assembler, layout_criterion, answer_criterion,max_layout_len, num_layers=1, decoder_dropout=0,**kwarg):

        super(end2endModuleNet, self).__init__()

        self.assembler = assembler
        self.layout_criterion = layout_criterion
        self.answer_criterion = answer_criterion


        ##initiate encoder and decoder
        myEncoder = EncoderRNN(num_vocab_txt, hidden_size, embed_dim_txt, num_layers)
        myDecoder = AttnDecoderRNN(hidden_size, num_vocab_nmn, embed_dim_nmn,
                                   max_decoder_len = max_layout_len,
                                   dropout_p=decoder_dropout, num_layers= num_layers,
                                   assembler_w=self.assembler.W, assembler_b=self.assembler.b,
                                   assembler_p=self.assembler.P, EOStoken=self.assembler.EOS_idx)

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

    def forward(self, input_txt_variable, input_text_seq_lens,
                input_images, input_answers,
                input_layout_variable,sample_token, policy_gradient_baseline=None,
                baseline_decay=None):

        batch_size = len(input_text_seq_lens)

        ##run attentionSeq2Seq
        myLayouts, myAttentions, neg_entropy, log_seq_prob = \
            self.mySeq2seq(input_txt_variable, input_text_seq_lens, input_layout_variable,sample_token)


        layout_loss = None
        if input_layout_variable is not None:
            layout_loss = torch.mean(-log_seq_prob)

        predicted_layouts = np.asarray(myLayouts.cpu().data.numpy())
        expr_list, expr_validity_array = self.assembler.assemble(predicted_layouts)

        ## group samples based on layout
        sample_groups_by_layout = unique_columns(predicted_layouts)

        ##run moduleNet
        answer_losses = None
        policy_gradient_losses = None
        avg_answer_loss =None
        total_loss = None
        updated_baseline = policy_gradient_baseline
        current_answer = np.zeros(batch_size)

        for sample_group in sample_groups_by_layout:
            if sample_group.shape == 0:
                continue

            first_in_group = sample_group[0]
            if expr_validity_array[first_in_group]:
                layout_exp = expr_list[first_in_group]

                if input_answers is None:
                    ith_answer_variable = None
                else:
                    ith_answer = input_answers[sample_group]
                    ith_answer_variable = Variable(torch.LongTensor(ith_answer))
                    ith_answer_variable = ith_answer_variable.cuda() if use_cuda else ith_answer_variable

                textAttention = myAttentions[sample_group, :]

                ith_image = input_images[sample_group, :, :, :]
                ith_images_variable = Variable(torch.FloatTensor(ith_image))
                ith_images_variable = ith_images_variable.cuda() if use_cuda else ith_images_variable

                ##image[batch_size, H_feat, W_feat, D_feat] ==> [batch_size, D_feat, W_feat, H_feat] for conv2d
                #ith_images_variable = ith_images_variable.permute(0, 3, 1, 2)

                ith_images_variable = ith_images_variable.contiguous()

                myAnswers = self.myModuleNet(input_image_variable=ith_images_variable,
                                        input_text_attention_variable=textAttention,
                                        target_answer_variable=ith_answer_variable,
                                        expr_list=layout_exp)
                current_answer[sample_group] = torch.topk(myAnswers, 1)[1].cpu().data.numpy()[:, 0]


                ##compute loss function only when answer is provided
                if ith_answer_variable is not None:
                    current_answer_loss = self.answer_criterion(myAnswers, ith_answer_variable)
                    sample_group_tensor = torch.cuda.LongTensor(sample_group) if use_cuda else torch.LongTensor(sample_group)
                
                    current_log_seq_prob = log_seq_prob[sample_group_tensor]
                    current_answer_loss_val = Variable(current_answer_loss.data,requires_grad=False)
                    tmp1 = current_answer_loss_val - policy_gradient_baseline
                    current_policy_gradient_loss = tmp1 * current_log_seq_prob

                    if answer_losses is None:
                        answer_losses = current_answer_loss
                        policy_gradient_losses = current_policy_gradient_loss
                    else:
                        answer_losses = torch.cat((answer_losses, current_answer_loss))
                        policy_gradient_losses = torch.cat((policy_gradient_losses, current_policy_gradient_loss))

        try:
            if input_answers is not None:
                total_loss, avg_answer_loss = self.layout_criterion(neg_entropy=neg_entropy,
                                                                            answer_loss=answer_losses,
                                                                            policy_gradient_losses=policy_gradient_losses,
                                                                            layout_loss=layout_loss)
                ##update layout policy baseline
                avg_sample_loss = torch.mean(answer_losses)
                avg_sample_loss_value = avg_sample_loss.cpu().data.numpy()[0]
                updated_baseline = policy_gradient_baseline + (1 - baseline_decay) * (
                avg_sample_loss_value - policy_gradient_baseline)

        except:
            print("sample_group = ", sample_group)
            print("neg_entropy=", neg_entropy)
            print("answer_losses=", answer_losses)
            print("policy_gradient_losses=", policy_gradient_losses)
            print("layout_loss=", layout_loss)
            sys.stdout.flush()
            sys.exit("Exception Occur")

                    
        
            

        return total_loss, avg_answer_loss, current_answer, predicted_layouts, expr_validity_array, updated_baseline






