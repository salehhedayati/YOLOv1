import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
import torch.utils

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotatoins = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transfrom = transform
        self.S = S
        self.B = B
        self.C = C
    
    def __len__(self):
        return len(self.annotatoins)
    
    def _getimages(self, index):
        label_path = os.path.join(self.label_dir, self.annotatoins.iloc[index, 1])
    
    