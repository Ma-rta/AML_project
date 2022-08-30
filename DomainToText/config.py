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


    batch_size=16
    val_every=20

    init_lr=0.00001 
    lr_decay_gamma = 0.1
    lr_decay_eval_count = 10

    weight_decay = 1e-6
    alpha = 0.8
    beta = 0.999
    epsilon = 1e-8

   