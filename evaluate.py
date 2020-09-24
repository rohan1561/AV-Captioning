import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import os
import argparse
import json
from pandas.io.json import json_normalize

import NLUtils
from cocoeval import suppress_stdout_stderr, COCOScorer
from dataloader import VideoAudioDataset
from models import MultimodalAtt
from tqdm import tqdm
import opts


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def eval(model, crit, loader, vocab, opt):
    model.eval()
    scorer = COCOScorer()
    ip_json = open(opt['input_json'])
    gt_dataframe = json_normalize(
        json.load(ip_json)['sentences'])
    ip_json.close()
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in tqdm(loader):
        # forward the model to get loss
        video_ids = data['video_ids']
        audio_fc2 = data['audio_fc2'].cuda()
        video_feat = data['video_feat'].cuda()
       
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(audio_fc2, video_feat,
                    mode='inference', opt=opt)

        sents = NLUtils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    return valid_score

def main(loader, vocab, opt, model=None):
    if model is None:
        vocab_size = len(vocab)
        model = MultimodalAtt(vocab_size, opt['max_len'],
                opt['dim_hidden'], opt['dim_word'])

        model = nn.DataParallel(model)

        if opt['beam']:
            bw = opt['beam_size']
            print(f'Using beam search with beam width = {bw}')
        model_path = opt['checkpoint_path']
        for i in os.listdir(model_path):
            if i.endswith('.pth'):
                print(i)
                path = os.path.join(model_path, i)
                model.load_state_dict(torch.load(path))
                crit = NLUtils.LanguageModelCriterion()

                eval(model, crit, loader, vocab, opt)
    else:
        '''
        Running from inside train.py
        '''
        crit = NLUtils.LanguageModelCriterion()
        scores = eval(model, crit, loader, vocab, opt)
        return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate options')
    parser.add_argument(
            "--gpu", type=str, default='0', help='GPU for evaluation')
    parser.add_argument(
            "--folder", type=str, default='save', help='folder of models')
    args = parser.parse_args()

    opt = json.load(open(os.path.join(args.folder, 'opt_info.json')))
    print(f'MODELS in the folder: {args.folder}')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    beam = input('beam? (y/n):')
    if beam == 'y':
        beam_size = int(input('beam size?:'))
        opt["beam"] = True
        opt["beam_size"] = beam_size
        batch_size = 1

    elif beam == 'n':
        opt['beam'] = False
        batch_size = opt['batch_size']
 
    mode = 'test'
    dataset = VideoAudioDataset(opt, mode)
    vocab = dataset.get_vocab()
    loader_val = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    main(loader_val, vocab, opt)

