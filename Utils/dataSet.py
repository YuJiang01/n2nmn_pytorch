import os
from torch.utils.data import Dataset
from Utils import text_processing
import numpy as np



class vqa_dataset(Dataset):
    def __init__(self,imdb_file, image_feat_directory, **data_params):
        super(vqa_dataset,self).__init__()
        if imdb_file.endswith('.npy'):
            imdb = np.load(imdb_file)
        else:
            raise TypeError('unknown imdb format.')

        self.imdb = imdb
        self.image_feat_directory = image_feat_directory
        self.data_params = data_params
        self.image_depth_first = data_params['image_depth_first']

        self.vocab_dict = text_processing.VocabDict(data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = (('answer' in self.imdb[0]) and (self.imdb[0]['answer'] is not None)) \
                           or (('valid_answers' in self.imdb[0]) and (self.imdb[0]['valid_answers'] is not None))
        self.load_gt_layout = ('gt_layout_tokens' in self.imdb[0]) and (self.imdb[0]['gt_layout_tokens'] is not None)
        if 'load_gt_layout' in data_params:
            self.load_gt_layout = data_params['load_gt_layout']
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(data_params['vocab_answer_file'])
        if not self.load_answer:
            print('imdb does not contain answers')
        if self.load_gt_layout:
            self.T_decoder = data_params['T_decoder']
            self.assembler = data_params['assembler']
            self.prune_filter_module = (data_params['prune_filter_module']
                                        if 'prune_filter_module' in data_params
                                        else False)
        else:
            print('imdb does not contain ground-truth layout')

        # load one feature map to peek its size
        image_file_name = os.path.basename(self.imdb[0]['feature_path'])
        image_feat_path = os.path.join(self.image_feat_directory,image_file_name)
        feats = np.load(image_feat_path)
        #self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):
        input_seq = np.zeros((self.T_encoder),np.int32)
        iminfo = self.imdb[idx]
        question_inds = [self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]
        seq_length = len(question_inds)
        input_seq[:seq_length] = question_inds
        image_file_name = os.path.basename(self.imdb[idx]['feature_path'])
        image_feat_path = os.path.join(self.image_feat_directory, image_file_name)
        image_feat =np.squeeze(np.load(image_feat_path), axis=0)
        if not self.image_depth_first:
            image_feat = np.transpose(image_feat, axes=(2, 0, 1))
        answer = None
        if self.load_answer:
            if 'answer' in iminfo:
                answer = iminfo['answer']
            elif 'valid_answers' in iminfo:
                valid_answers = iminfo['valid_answers']
                answer = np.random.choice(valid_answers)
            answer_idx = self.answer_dict.word2idx(answer)

        if self.load_gt_layout:
            gt_layout_tokens = iminfo['gt_layout_tokens']
            if self.prune_filter_module:
                # remove duplicated consequtive modules (only keeping one _Filter)
                for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                    if (gt_layout_tokens[n_t - 1] in {'_Filter', '_Find'}
                            and gt_layout_tokens[n_t] == '_Filter'):
                        gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
            gt_layout =np.array(self.assembler.module_list2tokens(
                gt_layout_tokens, self.T_decoder))

        sample = dict(input_seq_batch=input_seq,
                     seq_length_batch=seq_length,
                     image_feat_batch=image_feat)
        if self.load_answer:
            sample['answer_label_batch'] = answer_idx
        if self.load_gt_layout:
            sample['gt_layout_batch'] = gt_layout

        return sample