import numpy as np
import sys
from models.layout_assembler import Assembler
from .input_parameters import *


snapshot_dir = './exp_clevr/tfmodel/%s/' % exp_name

assembler = Assembler(vocab_layout_file)

def run_training(max_iter, dataset_trn):
    avg_accuracy = 0
    accuracy_decay = 0.99
    for n_iter, batch in enumerate(dataset_trn.batches()):
        if n_iter >= max_iter:
            break


        # Part 0 & 1: Run Convnet and generate module layout
        tokens, entropy_reg_val = None #sess.partial_run(h,
        '''(nmn3_model_trn.predicted_tokens, nmn3_model_trn.entropy_reg),
            feed_dict={input_seq_batch: batch['input_seq_batch'],
                       seq_length_batch: batch['seq_length_batch'],
                       image_feat_batch: batch['image_feat_batch'],
                       gt_layout_batch: batch['gt_layout_batch']})'''



        tokens = batch['gt_layout_batch']
        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = assembler.assemble(tokens)
        # all exprs should be valid (since they are ground-truth)
        assert(np.all(expr_validity_array))

        labels = batch['answer_label_batch']
        # Build TensorFlow Fold input for NMN
        expr_feed = None #compiler.build_feed_dict(expr_list)
        #expr_feed[expr_validity_batch] = None#expr_validity_array
        #expr_feed[answer_label_batch] = None #labels

        # Part 2: Run NMN and learning steps
        scores_val, avg_sample_loss_val, _ = None #sess.partial_run(
           # h, (scores, avg_sample_loss, train_step), feed_dict=expr_feed)

        # compute accuracy
        predictions = np.argmax(scores_val, axis=1)
        accuracy = np.mean(np.logical_and(expr_validity_array,
                                          predictions == labels))
        avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)
        validity = np.mean(expr_validity_array)

        # Add to TensorBoard summary
        if (n_iter+1) % log_interval == 0 or (n_iter+1) == max_iter:
            print("iter = %d\n\tloss = %f, accuracy (cur) = %f, "
                  "accuracy (avg) = %f, entropy = %f, validity = %f" %
                  (n_iter+1, avg_sample_loss_val, accuracy,
                   avg_accuracy, -entropy_reg_val, validity))
            sys.stdout.flush()

            summary = None
            '''sess.run(log_step_trn, {
                loss_ph: avg_sample_loss_val,
                entropy_ph: -entropy_reg_val,
                accuracy_ph: avg_accuracy,
                # baseline_ph: sess.run(baseline),
                validity_ph: validity})'''
            #log_writer.add_summary(summary, n_iter+1)

        # Save snapshot
        if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
            snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
            #snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
            print('snapshot saved to ' + snapshot_file)
