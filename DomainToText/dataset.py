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
