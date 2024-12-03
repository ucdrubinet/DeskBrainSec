from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class HeatmapDataset(Dataset):
    def __init__(self, tile_dir, row, col, stride=1):
        """
        Args:
            tile_dir (string): path to the folder where tiles are
            row (int): row index of the tile being operated
            col (int): column index of the tile being operated
            stride: stride of sliding 
        """
        self.tile_row = row
        self.tile_col = col
        self.tile_size = 256
        self.img_size = 1536
        self.stride = stride
        self.transform = transforms.Compose([
            # Normalize for pretrained weights
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        padding = 128
        # Create large image of ones to fill
        large_img = torch.ones(3, 3 * self.img_size, 3 * self.img_size)
        
        # fill large image with 3 x 3 window around tile at (row, col)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                img_path = f"{tile_dir}/{self.tile_row + i}/{self.tile_col + j}.jpg"
                # Attempt to load the neighboring tile
                try:
                    img = Image.open(img_path)
                    if self.transform:
                        img = self.transform(img)
                    
                    # Place the loaded tile in the correct position in the large image
                    large_img[:, 
                              (i + 1) * self.img_size : (i + 2) * self.img_size, 
                              (j + 1) * self.img_size : (j + 2) * self.img_size] = img
                # handle if tile does not exist (for corner and edge tiles)
                except FileNotFoundError:
                    pass
        # Crop to the central image with padding
        self.padding_img = large_img[
            :,
            self.img_size - padding : 2 * self.img_size + padding, 
            self.img_size - padding : 2 * self.img_size + padding
        ]
        # Calculate number of sliding window positions
        self.len = (self.img_size // self.stride) ** 2
        
    def __getitem__(self, index):

        row = (index * self.stride // self.img_size) * self.stride
        col = (index * self.stride % self.img_size)

        img = self.padding_img[:, row : row + self.tile_size, col : col + self.tile_size]        
    
        return img

    def __len__(self):
        return self.len
