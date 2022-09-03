import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from model.triplet_match.model import TripletMatch
from torch.utils.data import DataLoader
import numpy as np
#from PIL import Image
import random

from DomainToText.dataset import TrainDataset, EvalDataset
from DomainToText.evaluation import do_eval

def split(annotations_path, train_pctg=0.6, val_pctg=0.2, test_pctg=0.2):
    assert np.isclose(train_pctg + val_pctg + test_pctg, 1.0)
    train_txt, val_txt, test_txt = [], [], []
    _, visual_domains, _ = next(os.walk(annotations_path))
    for domain in visual_domains:
        _, categories, _ = next(os.walk(annotations_path + domain))
        for cat in categories:
            _, _, files = next(os.walk(annotations_path + domain + '/' + cat))
            N = len(files)
            Ntrain, Nval = np.floor(N * train_pctg), np.ceil(N * val_pctg)
            Nval += Ntrain
            for i, file in enumerate(files):
                if i < Ntrain: train_txt.append(domain + '/' + cat + '/' + file)
                elif i < Nval: val_txt.append(domain + '/' + cat + '/' + file)
                else: test_txt.append(domain + '/' + cat + '/' + file)
    return train_txt, val_txt, test_txt

static_idx_counter = 0
def batch_loader(model, trainset, batch_size):
        global static_idx_counter
        # Random selection
        imgs = []
        for img in trainset.images:
            imgs.append(torch.unsqueeze(trainset.img_transform(img), dim=0))
        t_images = torch.Tensor(len(trainset), 3, 224, 224)
        torch.cat(imgs, out=t_images)
        pos_indices = [(x + static_idx_counter) % len(trainset) for x in range(batch_size)]
        static_idx_counter = (static_idx_counter + batch_size) % len(trainset)
        pos_images = []
        pos_phrase = []
        neg_images = []
        neg_phrase = []
        for idx in pos_indices:
            pos_images.append(torch.unsqueeze(t_images[idx], dim=0).cpu())
            pos_phrase.append(trainset.descr[idx])
            pos_vd = trainset.visual_domain[idx]
            i = None
            while i is None or trainset.visual_domain[i] == pos_vd:
                i = random.choice(range(len(trainset)))
            neg_images.append(torch.unsqueeze(t_images[i], dim=0).cpu())
            i = None
            while i is None or trainset.visual_domain[i] == pos_vd:
                i = random.choice(range(len(trainset)))
            neg_phrase.append(trainset.descr[i])
        
        t_pos_images = torch.Tensor(batch_size, 3, 224, 224)
        t_neg_images = torch.Tensor(batch_size, 3, 224, 224)
        torch.cat(pos_images, out=t_pos_images)
        torch.cat(neg_images, out=t_neg_images)
        return t_pos_images, pos_phrase, t_neg_images, neg_phrase

def train(use_tensorboard=True):
    n_epoch=100
    batch_size=16
    val_epoch=20
    early_stop_eval_count = 40

    init_lr=0.00001
    lr_decay_gamma = 0.1
    lr_decay_eval_count = 10

    weight_decay = 1e-6
    alpha = 0.8
    beta = 0.999
    epsilon = 1e-8

    train_list, val_list, _ = split('/content/AML_project/datalabels/')
    if use_tensorboard: writer = SummaryWriter()

    trainset = TrainDataset(train_list)
    valset = EvalDataset(val_list)

    #data_loader = DataLoader(valset, batch_size, shuffle=True, drop_last=True, pin_memory=True)

    model = TripletMatch()
    model.cuda()

    if os.path.exists('finetuned/LAST_checkpoint.pth'):
        model.load_state_dict(torch.load('finetuned/LAST_checkpoint.pth'), strict=False)
    elif not os.path.exists('./finetuned'):
        os.mkdir('./finetuned')

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=(alpha, beta), eps=epsilon)

    best_eval_metric = 0
    best_eval_count = 0
    early_stop = False
    it = 0
    try:
      while epoch < n_epoch & and not early_stop:
        # Train
        pos_images, pos_phrase, neg_images, neg_phrase = batch_loader(model, trainset, batch_size)

        neg_img_loss, neg_sent_loss = model(pos_images.cuda(), pos_phrase, neg_images.cuda(), neg_phrase)
        loss = neg_img_loss + neg_sent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_tensorboard: writer.add_scalar('train/loss', loss, epoch)
        lr = optimizer.param_groups[0]['lr']
        if use_tensorboard: writer.add_scalar('train/lr', lr, epoch)

        # Validation
        if epoch % val_every == val_every-1:
                with torch.no_grad():
                    
                    i2p_result, p2i_result = do_eval(model, valset)
                    eval_metric = i2p_result + p2i_result
                   
                    if eval_metric > best_eval_metric:
                        print('best eval_metric',eval_metric, epoch)
                        best_eval_metric = eval_metric
                        best_eval_count = 0
                        torch.save(model.state_dict(), 'finetuned/BEST_checkpoint.pth')
                    else:
                        print('last eval_metric',eval_metric, epoch)
                        best_eval_count += 1
                        torch.save(model.state_dict(), 'finetuned/LAST_checkpoint.pth')

                    if best_eval_count % lr_decay_eval_count == 0 and best_eval_count > 0:
                        print('EVAL: lr decay triggered')
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_gamma

                    if best_eval_count % early_stop_eval_count == 0 and best_eval_count > 0:
                        print('EVAL: early stop triggered')
                        early_stop = True
                        break        
        epoch += 1

    except KeyboardInterrupt:
        writer.close()

if __name__ == '__main__':
    train()
