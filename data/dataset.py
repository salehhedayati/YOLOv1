import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transfrom = transform
        self.S = S
        self.B = B
        self.C = C
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
                
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transfrom:
            image, boxes = self.transfrom(image, boxes) # change the boxes after horizontal flip
            
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_labal, x, y, width, height = box.tolist()
            class_label = int(class_labal)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = self.S * width, self.S * height
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_cordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_cordinates
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix
            
# # test
# label_dir = "./archive/labels"
# img_dir = "./archive/images"
# csv_file = "./archive/8examples.csv"
# label_path = os.path.join(label_dir, annotations.iloc[1, 1])
# img_path = os.path.join(img_dir, annotations.iloc[1, 0])

# i_s, js, xs, ys = [], [], [], []
# for box in boxes:
#     class_labal, x, y, width, height = boxes[0].tolist()
#     class_label = int(class_labal)
#     i, j = int(7 * y), int(7 * x)
#     x_cell, y_cell = 7 * x - j, 7 * y - i
#     i_s.append(i)
#     js.append(j)
#     xs.append(x_cell)
#     ys.append(y_cell)
        
# width_cell, height_cell = 7 * width, 7 * height