import sys
sys.path.append('./')
import json
import os
import cv2
import yaml
import torch
from torch.utils.data import DataLoader
import PIL.Image as Image
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
import torch.functional as F 
from models import IntentPrediction
from dataset import JAADDataset

if __name__ == '__main__':
    # load config file
    config_file ='config_all.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    train_dataset = JAADDataset('E:/JAAD', train=True, dataset='jaad', configs=config)
    test_dataset = JAADDataset('E:/JAAD', train=False, dataset='jaad', configs=config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    
    model = IntentPrediction(config['model_opts'], config['train_opts'])
    breakpoint()
    model.train(dataloader=train_loader)
    model.eval(dataloader=test_loader)