import torch
from torch.utils.data import Dataset
import cv2
import os

class E2SDataset(Dataset):
    def __init__(self, dataset_root, augment=False):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        
        # find all images in the dataset root
        # the root should be like:
        # dataset_root/
        #   img1.jpg
        #   img2.jpg
        #   ...
        self.image_filenames = []
        for filename in os.listdir(dataset_root):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.image_filenames.append(os.path.join(dataset_root, filename))
        
        self.augment = augment
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)

        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_edge = image[:, :, :256]
        image_rgb = image[:, :, 256:]

        # Apply paired augmentation after split to keep RGB/semantic transforms identical.
        if self.augment:
            do_hflip = torch.rand(1).item() < 0.5
            do_vflip = torch.rand(1).item() < 0.1

            if do_hflip:
                image_rgb = torch.flip(image_rgb, dims=[2])
                image_edge = torch.flip(image_edge, dims=[2])

            if do_vflip:
                image_rgb = torch.flip(image_rgb, dims=[1])
                image_edge = torch.flip(image_edge, dims=[1])

        return image_edge, image_rgb