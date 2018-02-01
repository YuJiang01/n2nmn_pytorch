from __future__ import absolute_import, division, print_function

from train_model.run_training import *
from Utils.data_reader import DataReader

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

run_training(max_iter, data_reader_trn)





# Loss function

# The final per-sample loss, which is vqa loss for valid expr
# and invalid_expr_loss for invalid expr
final_loss_per_sample = None #softmax_loss_per_sample  # All exprs are valid

#avg_sample_loss = tf.reduce_mean(final_loss_per_sample)
#seq_likelihood_loss = tf.reduce_mean(-log_seq_prob)

#total_training_loss = seq_likelihood_loss + avg_sample_loss
#total_loss = total_training_loss + weight_decay * nmn3_model_trn.l2_reg

# Train with Adam
#solver = tf.train.AdamOptimizer()
#gradients = solver.compute_gradients(total_loss)

# Clip gradient by L2 norm
# gradients = gradients_part1+gradients_part2





