from os.path import join 
from os import listdir
import numpy as np
import pandas as pd
import h5py
import cv2
import glob
import torch.utils.data as udata
from torchvision.io import read_image
from PIL import Image
from utils import data_augmentation

class Dataset(udata.Dataset):
    def __init__(self, image_path, train=True, transform=None) -> None:
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            self.img_path = image_path
        else:
            self.img_path = image_path
        self.transform=transform
        
        
    def __len__(self):
        return len(listdir(self.img_path))
    
    def __getitem__(self, index):
        image = Image.open(join(self.img_path, listdir(self.img_path)[index]))
        if self.transform:
            image = self.transform(image)
        return image
    
