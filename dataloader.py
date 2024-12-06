import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, image_dir, size=None, transform=None):
        self.real_dir = os.path.join(image_dir, "0_real") 
        self.fake_dir = os.path.join(image_dir, "1_fake") 
        self.real_image_names = sorted(os.listdir(self.real_dir))
        self.fake_image_names = sorted(os.listdir(self.fake_dir))
        self.size = size
        if self.size:
            self.real_image_names = self.real_image_names[:self.size//2]
            self.fake_image_names = self.fake_image_names[:self.size//2]
        self.real_size = len(self.real_image_names)
        self.fake_size = len(self.fake_image_names)
        self.transform = transform

    def __len__(self):
        return self.real_size + self.fake_size

    def __getitem__(self, idx):
        if idx < self.real_size:
            img_path = os.path.join(self.real_dir, self.real_image_names[idx])
            label = torch.tensor(0.0)
        else:
            img_path = os.path.join(self.fake_dir, self.fake_image_names[idx-self.real_size])
            label = torch.tensor(1.0)
        image = self.rgb_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
