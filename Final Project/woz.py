from argparse import ArgumentParser
import yaml as yaml
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import TransfoXLConfig, TransfoXLModel, TransfoXLLMHeadModel
from transformers import XLNetTokenizer, XLNetModel, XLNetLMHeadModel
from datasets import load_metric
import torch
import sys
import json
import numpy as np
import re
from global_variables import *
from helper_functions import *
from custom_decoder import *

parser = ArgumentParser()
parser.add_argument('--config', default='./config_files/config.yaml', help='Config .yaml file to use for training')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tagging(dialogue):
    # may need to break string into words first
    dialogue_tokens = dialogue.split()
    #print("LENGTH OF STRING - > " + str(len(dialogue_tokens)))
    for i in range(len(dialogue_tokens)):
        dialogue_tokens[i] = re.sub(r'[0-9]{1,2}:[0-9]{2}', '<time>', dialogue_tokens[i])
        dialogue_tokens[i] = re.sub(r'[0-9]{5}[\s]*[0-9]{5}', '<phone-number>', dialogue_tokens[i])
        dialogue_tokens[i] = re.sub(r'[0-9]+', '<integer>', dialogue_tokens[i])
    #print("LENGTH OF STRING - > " + str(len(dialogue_tokens)))
    return ' '.join(dialogue_tokens)



def make_woz_datasets(bKnowledge):
    if bKnowledge:
        out_names = [output_data_path + '/woz.train_c.txt',
                     output_data_path + '/woz.valid_c.txt',
                     output_data_path + '/woz.test_c.txt']
    else:
        out_names = [output_data_path + '/woz.train_b.txt',
                     output_data_path + '/woz.valid_b.txt',
                     output_data_path + '/woz.test_b.txt',
                     output_data_path + '/woz.valid_a.txt',
                     output_data_path + '/woz.test_a.txt',
                     ]
    max_ins = [18, 2, 2, 2, 2]

    count = 0
    counts = []
    for dataset in range(len(out_names)):
        fout = open(out_names[dataset], 'wt')
        for dialog in range(1, max_ins[dataset], 1):
            file_name = input_train_data_path + '/dialogues_%03d.json' % dialog
            print(file_name)
            with open(file_name) as f:
                data = json.load(f)
            for dialogue in data:
                if len(dialogue['services']) == 1:
                    if dialogue['services'][0] == 'restaurant':
                        prev_speaker = ''
                        prev_utterance = ''
                        for turn in dialogue['turns']:
                            count = count + 1
                            speaker = turn['speaker']
                            utterance = turn['utterance']
                            utterance = tagging(utterance)
                            for frame in turn['frames']:
                                if frame['service'] == 'restaurant':
                                    knowledge = ''
                                    try:
                                        knowledge = '[KNOWLEDGE] '
                                        for slot in frame['slots']:
                                            temp = '%s [EQUAL] %s [SEP] ' % (slot['slot'], slot['value'])
                                            knowledge = knowledge + temp
                                    except:
                                        nothing = 1
                                    try:
                                        if len(knowledge) == 0:
                                            knowledge = '[KNOWLEDGE] '
                                        try:
                                            intent = frame['state']['active_intent']
                                            temp = '%s [EQUAL] %s [SEP] ' % ('active_intent', intent)
                                            knowledge = knowledge + temp
                                            slot_values = frame['state']['slot_values']
                                            for slot in slot_values:
                                                vals = slot_values[slot]
                                                for val in vals:
                                                    temp = '%s [EQUAL] %s [SEP] ' % (slot, val)
                                                    knowledge = knowledge + temp
                                        except:
                                            nothing = 1
                                    except:
                                        noting = 1

                            if len(prev_speaker) > 0:
                                if not bKnowledge:
                                    knowledge = ''
                                if dataset == 0:
                                    text = '[%s] %s %s [%s] %s [END]' % (prev_speaker,
                                                                         prev_utterance,
                                                                         knowledge,
                                                                         speaker,
                                                                         utterance)
                                else:
                                    text = '[%s] %s %s [%s] | %s [END]' % (prev_speaker,
                                                                           prev_utterance,
                                                                           knowledge,
                                                                           speaker,
                                                                           utterance)
                                fout.write('%s\n' % (text))
                                print(text)
                            prev_speaker = speaker
                            prev_utterance = utterance
        counts.append(count)
        count = 0
    print(counts)


def main():
    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    #######
    # Get the variables required from the yaml file

    generate_dir_if_not_exists(output_data_path)
    generate_dir_if_not_exists(input_data_path)

    make_woz_datasets(True)
    make_woz_datasets(False)

    for i in range(0, 5):
        for j in range(0, 4):

            gen_mode = j
            gen_labels = ['logits', 'greedy', 'beam', 'top-p']
            tuned_model = i

            if tuned_model == 0:
                path = output_data_path + '/gpt2_untrained'
                generate_dir_if_not_exists(path)
                tuned = 'gpt2'
                test_name = output_data_path + '/woz.test_a.txt'
            elif tuned_model == 1:
                path = output_data_path + '/gpt2_trained'
                generate_dir_if_not_exists(path)
                tuned = path
                test_name = output_data_path + '/woz.test_b.txt'
            elif tuned_model == 2:
                path = output_data_path + '/distilgpt2'
                generate_dir_if_not_exists(path)
                tuned = path
                test_name = output_data_path + '/woz.test_c.txt'
            elif tuned_model == 3:
                path = output_data_path + '/xlnet-base-cased'
                generate_dir_if_not_exists(path)
                tuned = path
                test_name = output_data_path + '/woz.test_c.txt'
            elif tuned_model == 4:
                path = output_data_path + '/roberta-base'
                generate_dir_if_not_exists(path)
                tuned = path
                test_name = output_data_path + '/woz.test_c.txt'
            '''
            elif tuned_model == 2:
                path = output_data_path + '/gpt2_fine_tuned'
                generate_dir_if_not_exists(path)
                tuned = path
                test_name = output_data_path + '/woz.test_b.txt'
            '''

            tokenizer = GPT2Tokenizer.from_pretrained(tuned)
            model = GPT2LMHeadModel.from_pretrained(tuned, pad_token_id=tokenizer.eos_token_id)
            model = model.cuda()
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print('total parameters = ', params)

            metric = load_metric("bleu")


            predicts = []
            refs = []
            best = []
            bleus = []
            max_len = 0
            total = 0
            with open(test_name, 'rt') as f:
                for line in f:
                    line = line.replace('\n', '')
                    text = line.split('|')
                    prompt = text[0].strip(' ')
                    in_ids = tokenizer.encode(prompt, add_special_tokens=True)
                    if len(in_ids) > max_len:
                        max_len = len(in_ids)
                    total = total + 1

            max_len = max_len + 32
            print('max_len: %d total: %d' % (max_len, total))

            obs = 0
            with open(test_name, 'rt') as f:
                for line in f:
                    line = line.replace('\n', '')
                    text = line.split('|')
                    prompt = text[0].strip(' ')
                    ref = text[1].strip(' ')
                    obs = obs + 1
                    in_ids = tokenizer.encode(prompt, add_special_tokens=True)

                    # LOGITS
                    if gen_mode == 0:
                        seq_len = 0
                        bDone = False
                        while not bDone:
                            input_ids = torch.tensor(in_ids).unsqueeze(0)
                            input_ids = input_ids.cuda()
                            outputs = model(input_ids, labels=input_ids)
                            decoded = []
                            for i in range(outputs[1].size(1)):
                                # Greedy ?
                                print("\n\n*******THE ARGMAX*******\n\n")
                                # HERE we need to implement the custom-made decoder
                                print(torch.argmax(outputs[1][0][i][:]))
                                decoded.append(torch.argmax(outputs[1][0][i][:]).item())
                            decoded = torch.tensor(decoded)
                            decoded = decoded.cuda()
                            in_ids.append(decoded[decoded.size(0) - 1].item())
                            input_ids = torch.tensor(in_ids).unsqueeze(0)
                            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                            tokens = text.split(' ')
                            if tokens[len(tokens) - 1] == '[END]':
                                bDone = True
                            if len(tokens) >= max_len:
                                bDone = True

                    if gen_mode == 1:
                        input_ids = torch.tensor(in_ids).unsqueeze(0)
                        input_ids = input_ids.cuda()
                        greedy = model.generate(input_ids, max_length=max_len)
                        text2 = tokenizer.decode(greedy[0], skip_special_tokens=False)
                        tokens = text2.split()

                    elif gen_mode == 2:
                        input_ids = torch.tensor(in_ids).unsqueeze(0)
                        input_ids = input_ids.cuda()
                        beam = model.generate(input_ids, max_length=max_len, num_beams=5, early_stopping=True)
                        text2 = tokenizer.decode(beam[0], skip_special_tokens=False)
                        tokens = text2.split()

                    elif gen_mode == 3:
                        input_ids = torch.tensor(in_ids).unsqueeze(0)
                        input_ids = input_ids.cuda()
                        top_p = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.90, top_k=0)
                        text2 = tokenizer.decode(top_p[0], skip_special_tokens=False)
                        tokens = text2.split()

                    if gen_mode == 4:
                        # Custom beam search decoder
                        input_ids = torch.tensor(in_ids).unsqueeze(0)
                        input_ids = input_ids.cuda()
                        beam = beam_search_decoder(input_ids, num_beams=3)
                        text2 = tokenizer.decode(beam[0], skip_special_tokens=False)
                        tokens = text2.split()

                    first = len(prompt.split(' '))
                    try:
                        pos_end = tokens.index('[END]')
                    except:
                        pos_end = len(tokens)
                    try:
                        pos_enduser = tokens.index('[END][USER]')
                    except:
                        pos_enduser = len(tokens)
                    try:
                        pos_endsystem = tokens.index('[END][SYSTEM]')
                    except:
                        pos_endsystem = len(tokens)
                    last = min(pos_end, pos_enduser, pos_endsystem, len(tokens))
                    predict = ' '.join(tokens[first:last])

                    predictions = [predict.split()]
                    references = [[ref.split()]]
                    predicts.append(predictions)
                    refs.append(references)

                    try:
                        results = metric.compute(predictions=predictions, references=references)
                        bleus.append(results['bleu'])
                    except:
                        print('ref:  ', ref)
                        print('pred: ', predict)
                    if results['bleu'] > 0.01:
                        print('ref:  ', ref)
                        print('pred: ', predict)
                        print('BLEU[%d]: %7.3f' % (obs, results['bleu']))
                        print(' ')
                        best.append(results['bleu'])

            print(best)
            if len(best) > 0:
                print('avg[%d]: %7.5f' % (len(best), sum(best) / float(len(best))))
                print(' ')

            results = metric.compute(predictions=predicts, references=refs)
            print('Final %s on %s BLEU: %7.3f % 7.3f' % (gen_labels[gen_mode], test_name, results['bleu'], sum(bleus) / 511.0))
            print(len(predicts), len(refs))
            print(' ')


if __name__ == "__main__":
    #sys.stdout = open('./files/data/metric_results.txt', 'w')
    main()
    #sys.stdout.close()
