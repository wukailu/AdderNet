# Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import math

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data', type=str, default='/home/kailu/.cache/torch/data')
parser.add_argument('--output_dir', type=str, default='/home/kailu/AdderNet/outputs/models/')
parser.add_argument('--cnn', help="if use cnn or ann, default use ann", action="store_true")
args = parser.parse_args()

if args.cnn:
    print("using CNN")
else:
    print("using ANN")

os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0

if args.dataset == "cifar10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                         transform=transform_train)
    data_test = CIFAR10(args.data,
                        train=False,
                        transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)
elif args.dataset == "PACS":
    from PACS import PACS
    d = PACS(["art_painting"], "art_painting", domain_info=False)
    data_train = d.gen_train_datasets()
    data_test = d.gen_val_datasets()
    data_train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=10, num_workers=0)
else:
    assert False

if args.dataset == "cifar10":
    if args.cnn:
        from models.resnet_cnn import cifar_resnet20
        net = cifar_resnet20().cuda()
    else:
        from models.resnet20 import resnet20
        net = resnet20().cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
elif args.dataset == "PACS":
    if args.cnn:
        from torchvision.models import resnet50
        net = resnet50(num_classes=7).cuda()
    else:
        from models.resnet50 import resnet50
        net = resnet50(num_classes=7).cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
else:
    assert False

criterion = torch.nn.CrossEntropyLoss().cuda()


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1 + math.cos(float(epoch) / 400 * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()


def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    epoch = 400
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net, args.output_dir + 'addernet' + ('_cnn' if args.cnn else '_ann') + '_' + args.dataset)


if __name__ == '__main__':
    main()
