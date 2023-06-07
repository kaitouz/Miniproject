import os

from PIL import Image, ImageOps
from glob import glob
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms

class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [ImageOps.exif_transpose(Image.open(p).convert('RGB')) for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return image
    

def MVTecDataloader(dataroot, batchSize, imageSize_h, imageSize_w, is_train=True):
    
    transform = transforms.Compose([
        transforms.Resize([imageSize_h, imageSize_w]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if is_train:
        image_list = sorted(glob(os.path.join(dataroot, 'train', 'good', '*.png')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False, drop_last=False)
        val_dataset = MVTecDataset(image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, drop_last=False)

        return train_loader, val_loader
    else:
        test_neg_image_list = sorted(glob(os.path.join(dataroot, 'test', 'good', '*.png')))
        test_pos_image_list = set(glob(os.path.join(dataroot, 'test', '*', '*.png'))) - set(test_neg_image_list)
        test_pos_image_list = sorted(list(test_pos_image_list))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

        return test_neg_loader, test_pos_loader