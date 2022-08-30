import os
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model.triplet_match.model import TripletMatch
from torch.utils.data import DataLoader
import numpy as np

from DomainToText.config import C as cfg
from DomainToText.dataset import TrainDataset, EvalDataset, random_batch
from DomainToText.evaluation import do_eval

def train_val_test_split(annotations_path, train_pctg=0.6, val_pctg=0.2, test_pctg=0.2):
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

def train(use_tensorboard=True):
 
    train_list, val_list, _ = train_val_test_split('/content/AML_project/datalabels/')
    if use_tensorboard: writer = SummaryWriter()

    trainset = TrainDataset(train_list)
    valset = EvalDataset(val_list)

    data_loader = DataLoader(valset, cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    model = TripletMatch()
    model.cuda()

    if os.path.exists('metric_learning/LAST_checkpoint.pth'):
        model.load_state_dict(torch.load('metric_learning/LAST_checkpoint.pth'), strict=False)
    elif not os.path.exists('./metric_learning'):
        os.mkdir('./metric_learning')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.INIT_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY, betas=(cfg.TRAIN.ADAM.ALPHA, cfg.TRAIN.ADAM.BETA),eps=cfg.TRAIN.ADAM.EPSILON)

    best_eval_metric = 0
    best_eval_count = 0
    it = 0
    while it<cfg.TRAIN.MAX_EPOCH:
        # Train
        pos_images, pos_phrase, neg_images, neg_phrase = random_batch(model, trainset, cfg.TRAIN.BATCH_SIZE)

        neg_img_loss, neg_sent_loss = model(pos_images.cuda(), pos_phrase, neg_images.cuda(), neg_phrase)
        loss = neg_img_loss + neg_sent_loss
        if use_tensorboard: writer.add_scalar('Loss/train', loss, it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Validation
        if it % cfg.TRAIN.EVAL_EVERY_EPOCH == cfg.TRAIN.EVAL_EVERY_EPOCH-1:
            with torch.no_grad():
                i2p_result, p2i_result = do_eval(model, valset)
                eval_metric = i2p_result + p2i_result
                   
                if eval_metric > best_eval_metric:
                    print('best eval_metric',eval_metric, it)
                    best_eval_metric = eval_metric
                    best_eval_count = 0
                    torch.save(model.state_dict(), 'metric_learning/BEST_checkpoint.pth')
                else:
                    print('last eval_metric',eval_metric, it)
                    best_eval_count += 1
                    torch.save(model.state_dict(), 'metric_learning/LAST_checkpoint.pth')

                if best_eval_count % cfg.TRAIN.LR_DECAY_EVAL_COUNT == 0 and best_eval_count > 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= cfg.TRAIN.LR_DECAY_GAMMA
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

    i2p_result, p2i_result = do_eval(model, testset)
    eval_metric = i2p_result + p2i_result

    print(f'mAP on test set: {eval_metric:.3f}')

if __name__ == '__main__':
    train()
    #test()
