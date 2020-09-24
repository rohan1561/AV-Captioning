import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from .Attention import Attention
from functools import cmp_to_key

def cmp(x, y):
    if x == y:
        return 0
    if x > y:
        return 1
    if x < y:
        return -1


def KeyL(x, y):
    return cmp(x[1], y[1])


class MultimodalAtt(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, 
        sos_id=1, eos_id=0):
        super(MultimodalAtt, self).__init__()

        #self.rnn_cell = nn.LSTM
        
        # dimension of the word embedding layer
        self.dim_word = dim_word
        # Output size would be a one-dimensional vector of the vocab base
        self.dim_output = vocab_size
        # The hidden size of the LSTM cells output. DEFAULT TO 256 
        self.dim_hidden = dim_hidden
        # Define the max length of either the generated sentence or training caption
        self.max_len = max_len
        # The ix in the vocab base for the <SOS> signal
        self.sos_id = sos_id
        # Same as above for <EOS>
        self.eos_id = eos_id

        # Define LSTM encoders
        self.fc2_encoder = nn.LSTM(
                input_size=128,
                hidden_size=self.dim_hidden,
                bidirectional=True,
                batch_first=True,
                )

        self.vid_encoder = nn.LSTM(
                input_size=512,
                hidden_size=self.dim_hidden,
                bidirectional=True,
                batch_first=True,
                )

        # DEFINE LINEAR LAYER TO PROJECT FC2 HIDDENSTATE TO C4 DIMENSIONS
        #self.linear_fc2 = nn.Linear(256, 512)

        # DEFINE ATTENTION LAYERS
        self.TemporalAttention_aud = Attention(2*self.dim_hidden)
        self.TemporalAttention_vid = Attention(2*self.dim_hidden)
        self.MultimodalAttention = Attention(2*self.dim_hidden)

        # DEFINE DECODER TO GENERATE CAPTION
        self.decoder = nn.LSTM(
                input_size=self.dim_word + 2*self.dim_hidden,
                hidden_size=2*self.dim_hidden,
                batch_first=True,
                )

        # LOADING THE PRE TRAINED EMBEDDINGS FROM FASTTEXT
        pretrained = torch.Tensor(np.load('/home/cxu-serve/p1/rohan27/'\
            'research/audiocaps/code2/helpers/fasttext_msrvtt.npy'))
        #self.embedding = nn.Embedding(self.dim_output, self.dim_word)
        self.embedding = nn.Embedding.from_pretrained(pretrained)

        # OUTPUT LAYER
        self.out = nn.Linear(512, self.dim_output)

    def forward(self, afc2, video_feat, target_variable=None, mode='train', opt={}):

        # GET THE INPUT SHAPES
        bs, seq_len, fc2_dim = afc2.shape 

        # ENCODE THE SOUND INPUTS
        aud_encoder_output, (aud_hidden_state, aud_cell_state) = self.fc2_encoder(afc2)
        aud_hidden_state = aud_hidden_state.view(1, bs, -1) # 1, bs, 512
        aud_cell_state = aud_cell_state.view(1, bs, -1) # 1, bs, 512

        # ENCODE THE VISUAL FEATURES FROM R3D
        vid_encoder_output, (vid_hidden_state, vid_cell_state) = self.vid_encoder(video_feat)
        vid_hidden_state = vid_hidden_state.view(1, bs, -1) # 1, bs, 512
        vid_cell_state = vid_cell_state.view(1, bs, -1) # 1, bs, 512

        if mode == 'train':
            print('Using AV and not AVVP')
        # CALCULATE THE HIDDEN AND CELL STATES FOR THE DECODER
        decoder_hidden = aud_hidden_state + vid_hidden_state 
        decoder_cell = aud_cell_state + vid_cell_state 

        # CALCULATE THE CONTEXT FOR THE DECODER INPUT USING MULTIMODAL ATTENTION
        aud_context = self.TemporalAttention_aud(decoder_hidden,
                aud_encoder_output) # bs, 1, 512
        vid_context = self.TemporalAttention_vid(decoder_hidden,
                vid_encoder_output) # bs, 1, 512
        full_context = torch.cat((aud_context, vid_context), dim=1) # bs, 2, 512
        context = self.MultimodalAttention(decoder_hidden, full_context)
        decoder_input = context

        seq_probs = list()
        seq_preds = list()
        if mode == 'train':
            for i in range(self.max_len - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) = \
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))

                aud_context = self.TemporalAttention_aud(decoder_hidden, aud_encoder_output)
                vid_context = self.TemporalAttention_vid(decoder_hidden, vid_encoder_output)
                full_context = torch.cat((aud_context, vid_context), dim=1)
                context = self.MultimodalAttention(decoder_hidden, full_context)
                decoder_input = context

                output = self.out(decoder_output)
                seq_probs.append(output)
            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'inference':
            current_words = self.embedding(torch.cuda.LongTensor([self.sos_id] * bs))

            for i in range(self.max_len-1):
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) =\
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))

                aud_context = self.TemporalAttention_aud(decoder_hidden, aud_encoder_output)
                vid_context = self.TemporalAttention_vid(decoder_hidden,
                    vid_encoder_output) # bs, 1, 512
                full_context = torch.cat((aud_context, vid_context), dim=1)
                context = self.MultimodalAttention(decoder_hidden, full_context)
                decoder_input = context

                logits = F.log_softmax(self.out(decoder_output).squeeze(1), dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        return seq_probs, seq_preds


