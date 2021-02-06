from abc import abstractmethod
from os import path
from typing import List

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PACSPATH = "/data/PACS"


class FullDatasetBase:
    mean: tuple
    std: tuple
    img_shape: tuple
    num_classes: int
    name: str

    def __init__(self, **kwargs):
        pass

    def gen_base_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]), None

    def gen_test_transforms(self):
        base, _ = self.gen_base_transforms()
        return base, _

    @abstractmethod
    def gen_train_transforms(self):
        return transforms.Compose([]), None

    @abstractmethod
    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    def sample_imgs(self) -> torch.Tensor:
        return torch.stack([torch.zeros(self.img_shape)] * 2)

    @staticmethod
    @abstractmethod
    def is_dataset_name(name: str):
        return name == "my_dataset"


class SinglePACS(Dataset):
    def __init__(self, subDatasetName, split, transform):
        self.filename = path.join(PACSPATH, 'split', subDatasetName + '_' + split + '.hdf5')
        self.transform = transform
        self.split = split
        domain_data = h5py.File(self.filename, 'r')
        self.pacs_imgs = domain_data.get('images')
        self.pacs_labels = np.array(domain_data.get('labels')) - 1  # Convert labels in the range(1,7) into (0,6)
        print('Domain ', subDatasetName)
        print('Image: ', self.pacs_imgs.shape, ' Labels: ', self.pacs_labels.shape,
              ' Out Classes: ', len(np.unique(self.pacs_labels)))
        unique, counts = np.unique(self.pacs_labels, return_counts=True)
        self.num_class = int(np.amax(unique) + 1)
        self.max_class_size = np.amax(counts)

    def __len__(self):
        return len(self.pacs_imgs)

    def __getitem__(self, index):
        curr_img = Image.fromarray(self.pacs_imgs[index, :, :, :].astype('uint8'), 'RGB')
        img = self.transform(curr_img)
        labels = int(self.pacs_labels[index])

        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        return img, labels, index


class DomainConcat(Dataset):
    def __init__(self, dataset_list: List[SinglePACS], domain_info=True):
        self.dataset_list = dataset_list
        self.len = sum([len(d) for d in dataset_list])
        self.domain_num = len(dataset_list)
        self.domain_info = domain_info

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        for idx, d in enumerate(self.dataset_list):
            if item < len(d):
                img, labels, index = d[item]
                domain = torch.eye(self.domain_num)[idx]
                if self.domain_info:
                    return img, labels, domain.long(), index
                else:
                    return img, labels
            else:
                item -= len(d)
        raise IndexError()


class PACS(FullDatasetBase):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_shape = (3, 224, 224)
    num_classes = 10
    name = "cifar10"

    def __init__(self, source_domain_list, target_domain, domain_info=True, **kwargs):
        self.source_name_list = source_domain_list
        self.target_name = target_domain
        self.domain_info = domain_info
        super().__init__(**kwargs)

    def gen_train_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            base_transforms
        ]), _

    def gen_test_transforms(self):
        base_transforms, _ = self.gen_base_transforms()
        return transforms.Compose([
            transforms.Resize((224, 224)),
            base_transforms
        ]), _

    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        if transform is None and target_transform is None:
            transform, target_transform = self.gen_train_transforms()
        return DomainConcat([SinglePACS(name, "train", transform) for name in self.source_name_list], self.domain_info)

    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        if transform is None and target_transform is None:
            transform, target_transform = self.gen_test_transforms()
        return DomainConcat([SinglePACS(name, "val", transform) for name in self.source_name_list], self.domain_info)

    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        if transform is None and target_transform is None:
            transform, target_transform = self.gen_test_transforms()
        return DomainConcat([SinglePACS(self.target_name, 'test', transform)], self.domain_info)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(PACS|pacs)", name)


if __name__ == "__main__":
    d = PACS(["art_painting", "cartoon", "photo"], "sketch")
    train_ds = d.gen_train_datasets()
    for i in range(10):
        print(train_ds[i])
