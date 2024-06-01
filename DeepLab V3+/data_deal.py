# use download dataset (Potsdam)
import os
import pathlib
import cv2

import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# class UsingDataset:
#     train_data = FashionMNIST(
#         root="./data/FashionMNIST",
#         train=True,
#         transform=transforms.ToTensor(),
#         download=True
#     )
#     train_loader = data.DataLoader(
#         dataset=train_data,
#         batch_size=64,
#         shuffle=True,
#         num_workers=3
#     )
#     test_data = FashionMNIST(
#         root="./data/FashionMNIST",
#         train=False,
#         transform=transforms.ToTensor(),
#         download=True
#     )
#     test_loader = data.DataLoader(
#         dataset=test_data,
#         batch_size=64,
#         shuffle=True,
#         num_workers=3
#     )


class UsingOwnData(data.Dataset):
    """
        Using folder images
    """

    def __init__(self, path):
        if not os.path.isdir(path):
            raise ValueError("input file_path is not a dir")
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*image*/*.jpg'))
        self.all_image_paths = [str(path) for path in all_image_paths]
        all_label_paths = list(data_root.glob('*label*/*.png'))
        self.all_image_labels = [str(path) for path in all_label_paths]
        # self.mean = np.array(mean).reshape((1, 1, 3))
        # self.std = np.array(std).reshape((1, 1, 3))

    def __getitem__(self, index):
        img = cv2.imread(self.all_image_paths[index])
        # img = cv2.resize(img, (224, 224))
        # img = img / 255
        # # img = (img - self.mean) / self.std
        # img = np.transpose(img, [2, 0, 1])
        transform_gy = transforms.ToTensor()  # 将PIL.Image转化为tensor，即归一化。
        transform_compose = transforms.Compose([
            transform_gy,
        ])
        img = transform_compose(img)
        label = cv2.imread((self.all_image_labels[index]))
        # label = label / 255
        # label = (label - self.mean) / self.std
        label = np.transpose(label, [2, 0, 1])[0, :, :]
        # labels = label.squeeze(1)
        # label = self.all_image_labels[index]
        label = torch.tensor(label, dtype=torch.long)
        img = img.to(torch.float32)
        # label = label.to(torch.float32)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


class WorkData(UsingOwnData):
    def __init__(self, path):
        super().__init__(path)
        if not os.path.isdir(path):
            raise ValueError("input file_path is not a dir")
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*.jpg'))
        self.all_image_paths = [str(path) for path in all_image_paths]

    def __getitem__(self, item):
        img = cv2.imread(self.all_image_paths[item])
        # img = cv2.resize(img, (224, 224))
        img = img / 255
        # img = (img - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        img = torch.tensor(img)
        img = img.to(torch.float32)
        return img

    def __len__(self):
        return len(self.all_image_paths)

