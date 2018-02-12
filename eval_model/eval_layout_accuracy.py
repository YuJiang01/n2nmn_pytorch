from __future__ import absolute_import, division, print_function
from eval_model.layout_evaluator import run_eval
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', required=True)
parser.add_argument('--snapshot_name', required=True)
parser.add_argument('--test_split', required=True)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()


exp_name = args.exp_name
snapshot_name = args.snapshot_name
tst_image_set = args.test_split


layout_accuracy, layout_correct_total, num_questions_total,answer_accuracy = run_eval(exp_name,snapshot_name,tst_image_set,print_log=True)


print('On split: %s' % tst_image_set)
print('\t layout accuracy = %f (%d / %d) answer_accuracy= %f' %
      (layout_accuracy, layout_correct_total, num_questions_total,answer_accuracy))