from torch.utils.data import Dataset
import torch
from torchvision import transforms
from abc import abstractmethod
from torch import nn, Tensor

IMAGENET100PATH = "/data/"
CIFAR100PATH = "~/.cache/torch/data"
CIFAR10PATH = "~/.cache/torch/data"
PACSPATH = "/data/PACS"

class ExampleDataset(Dataset):
    def __init__(self, length=233):
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return index


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


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, input: Tensor):
        # x should be in shape of [N, C, H, W]
        assert input.dim() == 4
        return input * self.std + self.mean


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, input: Tensor):
        # x should be in shape of [N, C, H, W]
        assert input.dim() == 4
        # print(input.device, self.mean.device)
        return (input - self.mean) / self.std


def load_attack(model, attack: str):
    import torchattacks
    if attack == 'PGD':
        return torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
    elif attack == 'CW':
        return torchattacks.CW(model, targeted=False, c=1, kappa=0, steps=1000, lr=0.01)
    elif attack == 'BIM':
        return torchattacks.BIM(model, eps=4 / 255, alpha=1 / 255, steps=0)
    elif attack == 'FGSM':
        return torchattacks.FGSM(model, eps=1 / 255)
    else:
        raise NotImplementedError()


class AdvAttack(torch.nn.Module):
    # Check example here https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Targeted%20PGD%20with%20Imagenet.ipynb
    def __init__(self, model, attack_name, mean=None, std=None, dataset=None, device=None):
        super().__init__()
        if dataset is not None:
            mean = dataset.mean
            std = dataset.std
        normed_model = nn.Sequential(Normalize(mean, std), model)
        normed_model.to(device)
        normed_model.eval()
        self.attack = load_attack(normed_model, attack_name)
        self.unnorm = UnNormalize(mean=mean, std=std)
        self.donorm = Normalize(mean=mean, std=std)

    def forward(self, images, labels):
        self.eval()
        assert images.dim() == 4
        return self.donorm(self.attack(self.unnorm(images), labels))
