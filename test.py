# 2020.01.10-Changed for testing AdderNets
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from adder import adder2d
from utils import AdvAttack
from feature_similarity_measurement import cka_loss

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='ImageNet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset', default="/cache/imagenet/val/")
parser.add_argument('--model_dir', type=str,
                    help='path to dataset', default="models/ResNet50-AdderNet.pth")
best_acc1 = 0
args, unparsed = parser.parse_known_args()


def uniform_quant(x, bits):
    min_val = x.min()
    delta = (x.max() - min_val) / (1 << bits)
    return ((x - min_val) / delta + 0.5).int() * delta + min_val


def low_pricision(model: torch.nn.Module, target_module, width):
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            for param in module.parameters():
                param.data = uniform_quant(param.data, width)


def weight_clip(model: torch.nn.Module, target_module, w_range):
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            for param in module.parameters():
                param.data = param.data.clamp(w_range[0], w_range[1])


def get_model(dataset, model_path):
    if dataset == 'cifar10':
        from models import resnet20
        model = resnet20.resnet20()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model.load_state_dict(torch.load(model_path))
        model = model.module
        # weight_clip(model, adder2d, [-10, 10])
        # low_pricision(model, adder2d, 4)
    elif dataset == 'cifar10_cnn':
        from models.resnet_cnn import cifar_resnet20
        model = cifar_resnet20(pretrained='cifar10')

        # Do quantilization
        # model.eval()
        # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # torch.quantization.fuse_modules(model, [['conv1', 'bn1']], inplace=True)
        # for module in model.modules():
        #     if isinstance(module, BasicBlock):
        #         torch.quantization.fuse_modules(module, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
        # for module in model.modules():
        #     if isinstance(module, torch.nn.Sequential) and len(module) == 2:
        #         torch.quantization.fuse_modules(module, ['0', '1'], inplace=True)
        # print(model)
        # model_fp32_prepared = torch.quantization.prepare(model)
        # low_pricision(model, torch.nn.Conv2d, 6)
    elif dataset == 'ImageNet':
        from models import resnet50
        model = resnet50.resnet50()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model.load_state_dict(torch.load(model_path))
        model = model.module
    else:
        assert False
    return model


def get_val_loader(dataset):
    if dataset == 'cifar10' or dataset == 'cifar10_cnn':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif dataset == 'ImageNet':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        assert False
    return val_loader


def main():
    # create model
    model = get_model(args.dataset, args.model_dir)
    # model_ann = get_model('cifar10_cnn', "")

    cudnn.benchmark = True

    # Data loading code
    val_loader = get_val_loader(args.dataset)

    # validate(val_loader, model_fp32_prepared, 'cpu')
    # model = torch.quantization.convert(model_fp32_prepared)
    # print("testing quant model")

    acc1 = validate(val_loader, model, 'cuda:0')
    # acc2 = validate(val_loader, model_ann, 'cuda:0')
    # get_similarity(val_loader, model, model_ann, 'cuda:0')


def get_similarity(val_loader, model1, model2, device):
    meters = [[AverageMeter() for i in range(10)] for j in range(10)]
    model1.eval().to(device)
    model2.eval().to(device)

    measure = cka_loss()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)

            # compute output
            f_list1, _ = model1(input, with_feature=True)
            f_list2, _ = model2(input, with_feature=True)

            # print([f.shape for f in f_list1])
            # print([f.shape for f in f_list2])
            for idx, f1 in enumerate(f_list1):
                for idy, f2 in enumerate(f_list2):
                    meters[idx][idy].update(measure(f1, f2).item(), input.size(0))

    ret = [[y.avg for y in x] for x in meters]
    for t in ret:
        print(t)


def validate(val_loader, model, device):
    top1 = AverageMeter()
    top5 = AverageMeter()

    attack = AdvAttack(model, 'FGSM', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                       device='cuda:0').cuda()

    model.eval().to(device)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            with torch.enable_grad():
                input = attack(input, target).cuda()
            assert input.grad_fn is None
            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
