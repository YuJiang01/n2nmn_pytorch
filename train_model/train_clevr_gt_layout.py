from __future__ import absolute_import, division, print_function
from Utils.data_reader import DataReader
import sys
from models.layout_assembler import Assembler
from train_model.input_parameters import *
from models.end2endModuleNet import *
from models.custom_loss import custom_loss

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



criterion_layout = custom_loss(lambda_entropy = 0.01)
criterion_answer = nn.CrossEntropyLoss(size_average=False,reduce=False)

myModel = end2endModuleNet(num_vocab_txt=num_vocab_txt, num_vocab_nmn=num_vocab_nmn, out_num_choices=num_choices,
                           embed_dim_nmn=embed_dim_nmn, embed_dim_txt=embed_dim_txt,
                           image_height=H_feat, image_width=W_feat, in_image_dim=D_feat,
                           hidden_size=hidden_size, assembler=assembler, layout_criterion=criterion_layout,
                           answer_criterion=criterion_answer, num_layers=num_layers, decoder_dropout=0)




myOptimizer = optim.Adam(myModel.parameters(), weight_decay=0, lr=learning_rate)


avg_accuracy = 0
accuracy_decay = 0.99
avg_layout_accuracy = 0
updated_baseline = np.log(28)

for i_iter, batch in enumerate(data_reader_trn.batches()):
    if i_iter >= max_iter:
        break

    _, n_sample = batch['input_seq_batch'].shape
    input_text_seq_lens = batch['seq_length_batch']
    input_text_seqs = batch['input_seq_batch']
    input_layouts = batch['gt_layout_batch']
    input_images = batch['image_feat_batch']
    input_answers = batch['answer_label_batch']


    n_correct_layout = 0
    n_correct_answer = 0

    input_txt_variable = Variable(torch.LongTensor(input_text_seqs))
    input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

    input_layout_variable = Variable(torch.LongTensor(input_layouts))
    input_layout_variable = input_layout_variable.cuda() if use_cuda else input_layout_variable

    
    myOptimizer.zero_grad()

    total_loss, myAnswer, predicted_layouts, expr_validity_array, updated_baseline \
        = myModel(input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                  input_answers=input_answers, input_images=input_images,policy_gradient_baseline =updated_baseline,
                  baseline_decay = baseline_decay
                  # input_layout_variable=input_layout_variable,
                  )


    total_loss.backward()
    torch.nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
    myOptimizer.step()

    layout_accuracy = np.mean(np.all(predicted_layouts == input_layouts, axis=0))
    avg_layout_accuracy += (1 - accuracy_decay) * (layout_accuracy - avg_layout_accuracy)

    accuracy = np.mean(np.logical_and(expr_validity_array, myAnswer == input_answers))
    avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
    validity = np.mean(expr_validity_array)

    if (i_iter + 1) % 20 == 0 :
        print("iter:", i_iter + 1,
              " cur_layout_accuracy:%.3f"% layout_accuracy, " avg_layout_accuracy:%.3f"% avg_layout_accuracy,
              " cur_ans_accuracy:%.4f"% accuracy, " avg_answer_accuracy:%.4f"% avg_accuracy,
              " validity:%.4f"%validity, "total loss:%.4f"%total_loss.data.cpu().numpy()[0])

        sys.stdout.flush()

    # Save snapshot
    if  (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
        model_snapshot_file = os.path.join(snapshot_dir, "model_%08d" % (i_iter + 1))
        torch.save(myModel, model_snapshot_file)
        print('snapshot saved to ' + model_snapshot_file )
        sys.stdout.flush()







