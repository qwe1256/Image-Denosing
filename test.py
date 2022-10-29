from os import environ
from os import mkdir
from os.path import join 
import os
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from math import ceil
from models import DnCNN
from dataset import  Dataset
from utils import *
from matplotlib import pyplot as plt
import cv2

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "0"



@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    dataset_test = Dataset(join(get_original_cwd(), cfg.dataset.test_dataset), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))   
        ]))

    loader_test = DataLoader(dataset=dataset_test, num_workers=cfg.dataset.num_workers, 
                              batch_size=cfg.dataset.batch_size, shuffle=False)

    device = torch.device('cuda:0')
    model = DnCNN(channels=cfg.model.num_channels, num_of_layers=cfg.model.num_of_layers)
    model.apply(weights_init_kaiming) 
    criterion = getattr(nn, cfg.model.criterion)()
    ckpt = torch.load(join(get_original_cwd(),cfg.test.ckpt))
    model.load_state_dict(ckpt)
    model.to(device)
    criterion.to(device)
    model.eval()

    psnr_test = 0
    for k, img_test in enumerate(loader_test, 0):
            img_test = img_test.to(device)
            noise = torch.Tensor(img_test.size()).to(device).normal_(mean=0, std=cfg.test.noiseL/255.)
            imgn_test = img_test + noise
            out_test = torch.clamp(imgn_test-model(imgn_test, is_train=False), 0., 1.)
            psnr_test += batch_PSNR(out_test, img_test, 1.)
    psnr_test /= ceil(len(dataset_test)/cfg.dataset.batch_size)
    print(psnr_test)

if __name__ == "__main__":
    main()