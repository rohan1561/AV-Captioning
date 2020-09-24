import torch
import torch.nn as nn
from dataloader import VideoAudioDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import os
import json 
from models import MultimodalAtt
from NLUtils import LanguageModelCriterion
import opts
import evaluate

def train(loader, loader_val, val_vocab, model, crit, optimizer, lr_scheduler, opt):
    model = nn.DataParallel(model)
    max_scores = 0
    for epoch in range(opt['epochs']):
        model.train()
        save_flag=True
        lr_scheduler.step()
        iteration = 0

        for data in loader:
            audio_fc2 = data['audio_fc2'].cuda()
            video_feat = data['video_feat'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            torch.cuda.synchronize()
            optimizer.zero_grad()
            
            seq_probs, _ = model(audio_fc2, video_feat, 
                    labels, 'train', opt=opt)

            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))

        scores = evaluate.main(loader_val, val_vocab, opt, model)
        scores = scores['Bleu_4']
        if scores > max_scores:
            max_scores = scores
            print(scores)
            model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write(f"model_%{epoch}, bleu4: {scores}\n")
            f.close()

def main(opt):
    dataset = VideoAudioDataset(opt, 'train')
    opt['vocab_size'] = dataset.get_vocab_size()
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)

    dataset_val = VideoAudioDataset(opt, 'test')
    loader_val = DataLoader(dataset_val, batch_size=opt['batch_size'], shuffle=True)
    val_vocab = dataset_val.get_vocab()

    model = MultimodalAtt(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], opt['dim_word']) 
    model = model.cuda()
    crit = LanguageModelCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        amsgrad=True)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(loader, loader_val, val_vocab, model, crit, optimizer, exp_lr_scheduler, opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)


