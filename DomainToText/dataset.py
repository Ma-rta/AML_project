import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

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
