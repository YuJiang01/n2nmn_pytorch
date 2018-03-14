import yaml
import argparse
import os
from models.layout_assembler import Assembler
from models.end2endModuleNet import *
from models.custom_loss import custom_loss
from global_variables.global_variables import *
from Utils.data_reader import DataReader
from torch import optim
from Utils.dataSet import vqa_dataset
from torch.utils.data import DataLoader



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config yaml file")
parser.add_argument("--out_dir",type=str, required=True, help="output directory")
args = parser.parse_args()

config_file= args.config
out_dir = args.out_dir

with open(config_file, 'r') as f:
    config = yaml.load(f)

torch.manual_seed(1)

##update config file with commandline arguments


def prepare_train_data_set(**data_cofig):
    data_root_dir = data_cofig['data_root_dir']
    vocab_layout_file = os.path.join(data_root_dir, data_cofig['vocab_layout_file'])
    assembler = Assembler(vocab_layout_file)
    imdb_file_trn = os.path.join(data_root_dir, 'imdb',data_cofig['imdb_file_trn'])
    image_feat_dir = os.path.join(data_root_dir,data_cofig['preprocess_model'],'train')
    vocab_question_file = os.path.join(data_root_dir,data_cofig['vocab_question_file'])
    vocab_answer_file = os.path.join(data_root_dir,data_cofig['vocab_answer_file'])
    prune_filter_module = data_cofig['prune_filter_module']
    N = data_cofig['N']
    T_encoder = data_cofig['T_encoder']
    T_decoder = data_cofig['T_decoder']
    image_depth_first = data_cofig['image_depth_first']

    vqa_train_dataset = vqa_dataset(imdb_file=imdb_file_trn, image_feat_directory=image_feat_dir, T_encoder=T_encoder,
                                    T_decoder=T_decoder,
                                    assembler=assembler,
                                    vocab_question_file=vocab_question_file,
                                    vocab_answer_file=vocab_answer_file,
                                    prune_filter_module=prune_filter_module,
                                    image_depth_first=image_depth_first)

    data_reader_trn = DataLoader(dataset=vqa_train_dataset, batch_size=N, shuffle=True)

    num_vocab_txt = vqa_train_dataset.vocab_dict.num_vocab
    num_vocab_nmn = len(assembler.module_names)
    num_choices = vqa_train_dataset.answer_dict.num_vocab

    return data_reader_trn, num_vocab_txt, num_choices,num_vocab_nmn, assembler


def prepare_model(num_vocab_txt, num_choices, num_vocab_nmn,assembler, **model_config):
    if model_config['model_type'] == model_type_gt_rl:
        myModel = torch.load(model_config['model_path'])
    else:
        criterion_layout = custom_loss(lambda_entropy= model_config['lambda_entropy'])
        criterion_answer = nn.CrossEntropyLoss(size_average=False, reduce=False)

        myModel = end2endModuleNet(num_vocab_txt=num_vocab_txt, num_vocab_nmn=num_vocab_nmn,
                                   out_num_choices=num_choices, assembler= assembler,
                                   layout_criterion=criterion_layout, answer_criterion=criterion_answer,
                                   max_layout_len=model_config['T_decoder'], **model_config)
        myModel = myModel.cuda() if use_cuda else myModel

    return myModel


data_reader_trn, num_vocab_txt, num_choices, num_vocab_nmn, assembler = prepare_train_data_set(**config['data'], **config['model'])
myModel = prepare_model(num_vocab_txt, num_choices, num_vocab_nmn, assembler, **config['model'])

training_parameters = config['training_parameters']
myOptimizer = optim.Adam(myModel.parameters(),
                         weight_decay=training_parameters['weight_decay'],
                         lr=training_parameters['learning_rate'])

model_type = config['model']['model_type']
avg_accuracy = 0
accuracy_decay = 0.99
avg_layout_accuracy = 0
updated_baseline = np.log(28)
max_iter = training_parameters['max_iter']
baseline_decay = training_parameters['baseline_decay']
max_grad_l2_norm = training_parameters['max_grad_l2_norm']
snapshot_interval = training_parameters['snapshot_interval']
snapshot_dir = os.path.join(config['output']['root_dir'],"tfmodel",config['output']['exp_name'])
os.makedirs(snapshot_dir, exist_ok=True)

i_iter = 0
for iepoch in range(100):
    print("iepoch = ", iepoch)
    if i_iter >= max_iter:
        break
    for i, batch in enumerate(data_reader_trn):
        n_sample,_ = batch['input_seq_batch'].shape
        input_text_seq_lens = batch['seq_length_batch'].cpu().numpy()
        input_text_seqs = np.transpose(batch['input_seq_batch'].cpu().numpy())
        input_layouts = np.transpose(batch['gt_layout_batch'].cpu().numpy())
        input_images = batch['image_feat_batch'].cpu().numpy()
        input_answers = batch['answer_label_batch'].cpu().numpy()

        n_correct_layout = 0
        n_correct_answer = 0

        input_txt_variable = Variable(torch.LongTensor(input_text_seqs))
        input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

        input_layout_variable = None
        decoder_sampling = True

        if model_type == model_type_gt:
            decoder_sampling = False
            input_layout_variable = Variable(torch.LongTensor(input_layouts))
            input_layout_variable = input_layout_variable.cuda() if use_cuda else input_layout_variable

        myOptimizer.zero_grad()

        total_loss, avg_answer_loss, myAnswer, predicted_layouts, expr_validity_array, updated_baseline \
            = myModel(input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                      input_answers=input_answers, input_images=input_images, policy_gradient_baseline=updated_baseline,
                      baseline_decay=baseline_decay, input_layout_variable=input_layout_variable,
                      sample_token=decoder_sampling
                      )

        if total_loss is not None:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
            myOptimizer.step()

        layout_accuracy = np.mean(np.all(predicted_layouts == input_layouts, axis=0))
        avg_layout_accuracy += (1 - accuracy_decay) * (layout_accuracy - avg_layout_accuracy)

        accuracy = np.mean(np.logical_and(expr_validity_array, myAnswer == input_answers))
        avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
        validity = np.mean(expr_validity_array)

        if (i_iter + 1) % 20 == 0:
            print("iter:", i_iter + 1,
                  " cur_layout_acc:%.3f" % layout_accuracy, " avg_layout_acc:%.3f" % avg_layout_accuracy,
                  " cur_ans_acc:%.4f" % accuracy, " avg_answer_acc:%.4f" % avg_accuracy,
                  "total loss:%.4f" % total_loss.data.cpu().numpy()[0],
                  "avg_answer_loss:%.4f" % avg_answer_loss.data.cpu().numpy()[0])

            sys.stdout.flush()

        # Save snapshot
        if (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
            model_snapshot_file = os.path.join(snapshot_dir, "model_%08d" % (i_iter + 1))
            torch.save(myModel, model_snapshot_file)
            print('snapshot saved to ' + model_snapshot_file)
            sys.stdout.flush()
        i_iter += 1

