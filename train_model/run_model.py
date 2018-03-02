from train_model.input_parameters import *
from global_variables.global_variables import *
from torch.autograd import Variable

def run_model(data_reader_trn, myModel, isTraining, myOptimizer=None,
              max_iter=None, baseline_decay=None, use_gt_layput=False,
              max_grad_l2_norm=None):

    avg_accuracy = 0
    accuracy_decay = 0.99
    avg_layout_accuracy = 0
    updated_baseline = np.log(28)

    n_correct_layout_total = 0
    n_correct_answer_total = 0
    n_questions_total = 0

    for i_iter, batch in enumerate(data_reader_trn.batches()):
        if max_iter is not None and i_iter >= max_iter:
            break

        n_correct_layout = 0
        n_correct_answer = 0
        _, n_questions = batch['input_seq_batch'].shape
        n_questions_total += n_questions

        input_text_seq_lens = batch['seq_length_batch']
        input_text_seqs = batch['input_seq_batch']
        input_layouts = batch['gt_layout_batch']
        input_images = batch['image_feat_batch']
        input_answers = batch['answer_label_batch']


        input_txt_variable = Variable(torch.LongTensor(input_text_seqs))
        input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

        if isTraining and use_gt_layput:
            input_layout_variable = Variable(torch.LongTensor(input_layouts))
            input_layout_variable = input_layout_variable.cuda() if use_cuda else input_layout_variable
        else:
            input_layout_variable = None

        if myOptimizer is not None:
            myOptimizer.zero_grad()

        total_loss, avg_answer_loss, myAnswer, predicted_layouts, expr_validity_array, updated_baseline \
            = myModel(input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                      input_answers=input_answers, input_images=input_images, policy_gradient_baseline=updated_baseline,
                      baseline_decay=baseline_decay, input_layout_variable=input_layout_variable)


        if input_answers  is not None:
            n_correct_answer = np.sum(np.logical_and(expr_validity_array, myAnswer == input_answers))


        if input_layouts is not None:
            n_correct_layout = np.sum(np.all(predicted_layouts == input_layouts, axis=0))


        ##compute current accuracy and accumulated average accuracy for training step:
        if isTraining:
            layout_accuracy = n_correct_layout/n_questions
            answer_accuracy = n_correct_answer/n_questions
            avg_accuracy += (1 - accuracy_decay) * (answer_accuracy - avg_accuracy)
            avg_layout_accuracy += (1 - accuracy_decay) * (layout_accuracy - avg_layout_accuracy)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
            myOptimizer.step()

            if (i_iter + 1) % 20 == 0:
                print("iter:", i_iter + 1,
                      " cur_layout_acc:%.3f" % layout_accuracy, " avg_layout_acc:%.3f" % avg_layout_accuracy,
                      " cur_ans_acc:%.4f" % answer_accuracy, " avg_answer_acc:%.4f" % avg_accuracy,
                      "total loss:%.4f" % total_loss.data.cpu().numpy()[0],
                      "avg_answer_loss:%.4f" % avg_answer_loss.data.cpu().numpy()[0])

                sys.stdout.flush()

            # Save snapshot
            if (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
                model_snapshot_file = os.path.join(snapshot_dir, "model_%08d" % (i_iter + 1))
                torch.save(myModel, model_snapshot_file)
                print('snapshot saved to ' + model_snapshot_file)
                sys.stdout.flush()

        else:
            n_correct_answer_total += n_correct_answer
            n_correct_layout_total += n_correct_layout

    if not isTraining:
        return layout_accuracy, layout_correct_total, num_questions_total, answer_accuracy