import numpy as np
import json
import os

import sys
from Utils import text_processing
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",type=str, required=True, help="directory for data ")
parser.add_argument("--out_dir",type=str, required=True, help="output directory for json files")
args = parser.parse_args()
data_dir = args.data_dir
out_dir = args.out_dir

question_file = 'CLEVR_%s_questions_gt_layout.json'

def build_imdb(image_set):
    print('building imdb %s' % image_set)
    question_file_name = (question_file % image_set)
    question_file_path = os.path.join(data_dir, question_file_name )
    with open(question_file_path) as f:
        questions = json.load(f)
    imdb = [None]*len(questions)
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_name = q['image_filename'].split('.')[0]
        feature_name = image_name + '.npy'
        question_str = q['question']
        question_tokens = text_processing.tokenize(question_str)
        gt_layout_tokens = None
        if 'gt_layout' in q:
            gt_layout_tokens = q['gt_layout']
        answer = None
        if 'answer' in q:
            answer = q['answer']

        iminfo = dict(image_name=image_name,
                      feature_path=feature_name,
                      question_str=question_str,
                      question_tokens=question_tokens,
                      gt_layout_tokens=gt_layout_tokens,
                      answer=answer)
        imdb[n_q] = iminfo
    return imdb


imdb_trn = build_imdb('train')
imdb_val = build_imdb('val')
imdb_tst = build_imdb('test')

os.makedirs('out_dir', exist_ok=True)

out_trn = os.path.join(out_dir, 'imdb_trn.npy')
out_val = os.path.join(out_dir, 'imdb_val.npy')
out_tst = os.path.join(out_dir, 'imdb_tst.npy')

np.save(out_trn, np.array(imdb_trn))
np.save(out_val, np.array(imdb_val))
np.save(out_tst, np.array(imdb_tst))
