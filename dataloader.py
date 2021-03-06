import json
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoAudioDataset(Dataset):
    def __init__(self, opt, mode):
        super(VideoAudioDataset, self).__init__()
        self.mode = mode

        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        self.closest_audio = info['aud_closest']

        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))
        self.feats_dir = opt['output_dir']
        print('load feats from %s' % (self.feats_dir))
        self.max_len = opt['max_len']
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])

        audio_fc2 = np.load(os.path.join(self.feats_dir,
            'audios_msrvtt/features/fc2', f'video{ix}.npy'))
        video_feat = np.load(os.path.join(self.feats_dir, 
            'vid_feats/features/R3D', f'video{ix}.npy'))

        mask = np.zeros(self.max_len)
        label = np.zeros(self.max_len)
        captions = self.captions[f'video{ix}']['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        cap_ix = random.randint(0, len(captions)-1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0])+1]=1

        data = dict()
        data['audio_fc2'] = torch.from_numpy(audio_fc2).type(torch.FloatTensor)
        data['video_feat'] = torch.from_numpy(video_feat).type(torch.FloatTensor)

        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['video_ids'] = f'video{ix}'
        data['gts'] = torch.from_numpy(gts).long()
        return data

    def __len__(self):
        return len(self.splits[self.mode])
    
    def get_vocab_size(self):
        return len(self.ix_to_word)
    
    def get_vocab(self):
        return self.ix_to_word
    
        
