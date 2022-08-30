import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from model.triplet_match.model import TripletMatch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import random

import dataset 
import eval


def train(use_tensorboard=True):
    batch_size=16
    val_every=20

    init_lr=0.00001
    lr_decay_gamma = 0.1
    lr_decay_eval_count = 10

    weight_decay = 1e-6
    alpha = 0.8
    beta = 0.999
    epsilon = 1e-8

    train_list, val_list, _ = train_val_test_split('/content/AML_project/datalabels/')
    if use_tensorboard: writer = SummaryWriter()

    trainset = TrainDataset(train_list)
    valset = EvalDataset(val_list)

    data_loader = DataLoader(valset, batch_size, shuffle=True, drop_last=True, pin_memory=True)

    model = TripletMatch()
    model.cuda()

    if os.path.exists('metric_learning/LAST_checkpoint.pth'):
        model.load_state_dict(torch.load('metric_learning/LAST_checkpoint.pth'), strict=False)
    elif not os.path.exists('./metric_learning'):
        os.mkdir('./metric_learning')

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, betas=(alpha, beta), eps=epsilon)

    best_eval_metric = 0
    best_eval_count = 0
    it = 0
    while it<250:
        # Train
        pos_images, pos_phrase, neg_images, neg_phrase = generate_minibatch(model, trainset, batch_size)

        neg_img_loss, neg_sent_loss = model(pos_images.cuda(), pos_phrase, neg_images.cuda(), neg_phrase)
        loss = neg_img_loss + neg_sent_loss
        if use_tensorboard: writer.add_scalar('Loss/train', loss, it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Validation
        if it % val_every == val_every-1:
                i2p_result, p2i_result = do_eval(model, valset)
                eval_metric = i2p_result + p2i_result
                   
                if eval_metric > best_eval_metric:
                    print('best eval_metric',eval_metric)
                    best_eval_metric = eval_metric
                    best_eval_count = 0
                    torch.save(model.state_dict(), 'metric_learning/BEST_checkpoint.pth')
                else:
                    print('last eval_metric',eval_metric)
                    best_eval_count += 1
                    torch.save(model.state_dict(), 'metric_learning/LAST_checkpoint.pth')

                if best_eval_count % lr_decay_eval_count == 0 and best_eval_count > 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_decay_gamma
                        if use_tensorboard: writer.add_scalar('LR/train', param_group['lr'], it)
           
        it += 1
    writer.close()

def test():
    _, _, test_list = train_val_test_split('/content/AML_project/datalabels/')
    testset = EvalDataset(test_list)

    model = TripletMatch()
    model.cuda()
    model.eval()

    if os.path.exists('/metric_learning/BEST_checkpoint.pth'):
        model.load_state_dict(torch.load('/metric_learning/BEST_checkpoint.pth'), strict=False)

    with torch.no_grad():
        out_img = model.img_encoder(testset.t_images.cuda()).cpu().numpy()
        out_txt = model.lang_encoder(testset.descr).cpu().numpy()
        match_scores = np.zeros((len(testset), len(testset)))
        gt_matrix = np.eye(len(testset))
        for i, img in enumerate(out_img):
            for j, phr in enumerate(out_txt):
                match_scores[i,j] = - np.sum(np.power(img - phr, 2)) # l2_s

        mAP_i2p = compute_mAP(match_scores, gt_matrix, mode='i2p')
        mAP_p2i = compute_mAP(match_scores, gt_matrix, mode='p2i') 

        eval_metric = mAP_i2p + mAP_p2i
        print(f'mAP on test set: {eval_metric:.3f}')

if __name__ == '__main__':
    train()
    #test()
