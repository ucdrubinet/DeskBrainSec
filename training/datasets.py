import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class WMGMDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        single_img_name = os.path.join(self.img_path, self.data.iloc[index]['filename'])
        label = self.data.iloc[index]['label']
        image = Image.open(os.path.join(self.img_path, single_img_name))

        if self.transform:
            image = self.transform(image)

        return image, label

# Credit: https://github.com/keiserlab/plaquebox-paper/blob/master/2.1)%20CNN%20Models%20-%20Model%20Training%20and%20Development.ipynb

class PlaqueDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        c=torch.Tensor(self.data_info.loc[:,'cored'])
        d=torch.Tensor(self.data_info.loc[:,'diffuse'])
        a=torch.Tensor(self.data_info.loc[:,'CAA'])
        c=c.view(c.shape[0],1)
        d=d.view(d.shape[0],1)
        a=a.view(a.shape[0],1)
        self.raw_labels = torch.cat([c,d,a], dim=1)
        self.labels = (torch.cat([c,d,a], dim=1)>0.99).type(torch.FloatTensor)

    def __getitem__(self, index):
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]
        raw_label = self.raw_labels[index]
        # Get image name from the pandas df
        single_image_name = str(self.data_info.loc[index,'imagename'])
        # Open image
        img = Image.open(os.path.join(self.img_path, single_image_name))
        # Transform image to tensor
        if self.transform:
            img = self.transform(img)
        # Return image and the label
        return img, single_image_label

    def __len__(self):
        return len(self.data_info.index)