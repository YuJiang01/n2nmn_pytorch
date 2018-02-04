import sys
import os

from eval_model.layout_evaluator import run_eval

exp_name = 'clevr_gt_layout'
dataSplitSet = 'val'
out_file = "layout_learning_on_eval_dataset.txt"

eval_results = []


with open(out_file, 'w') as f:
    for i_iter in range(500):
        snapshot_name = '%08d' % i_iter
        snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, snapshot_name)
        if os.path.exists(snapshot_file):
            accuracy,_, total = run_eval(exp_name, snapshot_name, dataSplitSet)
            eval_results.append((i_iter,accuracy,total))
            print("iter:", i_iter,"\taccuracy:", accuracy, "\ttotal:", total)
            sys.stdout.flush()
            print("iter:", i_iter, "\taccuracy:", accuracy, "\ttotal:", total,file=f)

