from os import environ
from os import mkdir
from os.path import join 
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from math import ceil
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from matplotlib import pyplot as plt


environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "0"


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if cfg.dataset.preprocess:
        if cfg.train.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if cfg.train.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    
    # Load dataset
    print('Loading dataset ...\n')
    
    dataset_train = Dataset(join(get_original_cwd(), cfg.dataset.train_dataset), train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ]))
    
    dataset_val = Dataset(join(get_original_cwd(), cfg.dataset.val_dataset), False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ]))
    loader_train = DataLoader(dataset=dataset_train, num_workers=cfg.dataset.num_workers, 
                              batch_size=cfg.dataset.batch_size, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=cfg.dataset.num_workers, 
                              batch_size=cfg.dataset.batch_size, shuffle=False)
    
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    model = DnCNN(channels=cfg.model.num_channels, num_of_layers=cfg.model.num_of_layers)
    model.apply(weights_init_kaiming) 
    criterion = getattr(nn, cfg.model.criterion)()
    # Move to GPU
    device = torch.device('cuda:0')
    model.to(device)
    criterion.to(device)
    # Optimizer
    optimizer = getattr(optim, cfg.model.optim.type)(model.parameters(), lr=cfg.model.optim.lr)
    # training
    mkdir('logs')
    writer = SummaryWriter('logs')
    step = 0
    for epoch in range(cfg.train.epoch):
        if epoch < (cfg.train.lr_decay):
            current_lr = cfg.model.optim.lr
        else:
            current_lr = (cfg.model.optim.lr)/ 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, img_train in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = img_train.to(device)
            if cfg.train.mode == 'S':
                noise = torch.Tensor(img_train.size()).to(device).normal_(mean=0, std=cfg.train.noiseL/255.)
            if cfg.train.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(cfg.train.noiseLB[0], cfg.train.noiseLB[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.Tensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            out_train = model(imgn_train)
            loss = criterion(out_train, noise)
            loss.backward()
            optimizer.step()
            # results
            out_train = torch.clamp(imgn_train-model(imgn_train, is_train=False), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR_on_training_data', psnr_train, step)
            step += 1
        ## the end of each epoch
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train, is_train=False), 0., 1.)
        Img = utils.make_grid(img_train, nrow=8, normalize=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean_image', Img, epoch)
        writer.add_image('noisy_image', Imgn, epoch)
        writer.add_image('reconstructed_image', Irecon, epoch)
        del out_train, img_train, imgn_train
        # validate
        model.eval()
        psnr_val = 0
        for k, img_val in enumerate(loader_val, 0):
            img_val = img_val.to(device)
            noise = torch.Tensor(img_val.size()).to(device).normal_(mean=0, std=cfg.val.noiseL/255.)
            imgn_val = img_val + noise
            out_val = torch.clamp(imgn_val-model(imgn_val, is_train=False), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= ceil(len(dataset_val)/cfg.dataset.batch_size)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR_on_validation_data', psnr_val, epoch)

        # save model
        torch.save(model.state_dict(), join('logs', f'net_{epoch}.pth'))

if __name__ == "__main__":
    main()
