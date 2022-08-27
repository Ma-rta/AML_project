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

def build_transforms(is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

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


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list, images_path='/content/AML_project/PACS/kfold/', annotations_path='/content/AML_project/datalabels/'):
        self.img_transform = build_transforms()
        self.images = []
        self.descr = []
        self.visual_domain = []
        for path in sample_list:
            if 'sketch' in path: self.images.append(Image.open(images_path + path.split('.txt')[0]+('.png')).convert('RGB'))
            else: self.images.append(Image.open(images_path + path.split('.txt')[0]+('.jpg')).convert('RGB'))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.visual_domain.append(0)
            elif 'cartoon' in path: self.visual_domain.append(1)
            elif 'photo' in path: self.visual_domain.append(2)
            elif 'sketch' in path: self.visual_domain.append(3)

    def __len__(self):
        return len(self.images)

class EvalDataset(torch.utils.data.Dataset): # both for validation, test -> return (pos_img, pos_txt)
    def __init__(self, sample_list,images_path='/content/AML_project/PACS/kfold/', annotations_path='/content/AML_project/datalabels/'):
        self.img_transformer = build_transforms(is_train=False)
        self.indexes = {'art_painting': [], 'cartoon': [], 'photo': [], 'sketch': []}
        self.images = []
        self.descr = []
        idx = 0
        for path in sample_list:
            if 'sketch' in path:self.images.append(torch.unsqueeze(self.img_transformer(Image.open(images_path + path.split('.txt')[0]+('.png')).convert('RGB')), dim=0))
            else: self.images.append(torch.unsqueeze(self.img_transformer(Image.open(images_path + path.split('.txt')[0]+('.jpg')).convert('RGB')), dim=0))
            with open(annotations_path + path, 'r') as f:
                self.descr.append(f.read())
            if 'art_painting' in path: self.indexes['art_painting'].append(idx)
            elif 'cartoon' in path: self.indexes['cartoon'].append(idx)
            elif 'photo' in path: self.indexes['photo'].append(idx)
            elif 'sketch' in path: self.indexes['sketch'].append(idx)
            idx += 1
        self.t_images = torch.Tensor(len(sample_list), 3, 224, 224)
        torch.cat(self.images, out=self.t_images)

    def __getitem__(self, index):
        img, txt = self.images[index], self.descr[index]
        if index in self.indexes['art_painting']: label = 'art_painting'
        elif index in self.indexes['cartoon']: label = 'cartoon'
        elif index in self.indexes['photo']: label = 'photo'
        elif index in self.indexes['sketch']: label = 'sketch'
        return (img, txt, label)

    def __len__(self):
       return len(self.images)

static_idx_counter = 0
def generate_minibatch(model, trainset, batch_size):
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
    batch_size=16
    mode='batch_hard'
    val_every=20

    #init_lr=0.000002 # for hard-negative
    init_lr=0.00001 # for batch-hard
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
    try:
      while it<20:
        # Train
        pos_images, pos_phrase, neg_images, neg_phrase = generate_minibatch(model, trainset, batch_size)

        neg_img_loss, neg_sent_loss = model(pos_images.cuda(), pos_phrase, neg_images.cuda(), neg_phrase)
        loss = neg_img_loss + neg_sent_loss
        print('Loss',loss, it)
        print(use_tensorboard)
        if use_tensorboard: writer.add_scalar('Loss/train', loss, it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % val_every == val_every-1:
          print('fine train')

        it += 1
    except KeyboardInterrupt:
      writer.close()

    
  
    
if __name__ == '__main__':
  train()