import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import os 

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--dataroot', default = './datasets', type = str)
parser.add_argument('--runs', default=3, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--seed', default=20, type=int)
args = parser.parse_args()

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

order_fn = np.nanargmax

def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

def main():

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.dataroot, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.dataroot, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    archs = ["alexnet", "resnet18", "resnet34", "resnet50", "vgg11", "vgg13", "densenet121", "squeezenet1_0", "squeezenet1_1"]
    # archs = ["resnet18", "resnet34", "resnet50"]

    for run in range(args.runs):
        scores = []
        for arch in archs:
            net = getattr(torchvision.models, arch)()
            if "classifier" in net._modules.keys():
                if isinstance(net.classifier, nn.Sequential):
                    if isinstance(net.classifier[-1], nn.Linear):
                        net.classifier[-1] = torch.nn.Linear(in_features=net.classifier[-1].in_features, out_features=10)
                else:
                    if isinstance(net.classifier, nn.Linear):
                        net.classifier = torch.nn.Linear(in_features=net.classifier.in_features, out_features=10)
            
            elif "fc" in net._modules.keys():
                if isinstance(net.fc, nn.Sequential):
                    if isinstance(net.fc[-1], nn.Linear):
                        net.fc[-1] = torch.nn.Linear(in_features=net.fc[-1].in_features, out_features=10)
                else:
                    if isinstance(net.fc, nn.Linear):
                        net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=10)
                

            # net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=10)
            # models.append(net())
            data_iterator = iter(trainloader)
            x, target     = next(data_iterator)
            x, target = x.cuda(), target.cuda()
            net = net.cuda()
            # net.eval()
            net.train()
            jacobs, labels= get_batch_jacobian(net, x, target)
            jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
            try:
                s = eval_score(jacobs, labels)
            except Exception as e:
                print(e)
                s = np.nan
            scores.append(s)
        print(scores)

        best_arch = archs[order_fn(scores)]
        print(best_arch)

if __name__ == "__main__": 
    
    if not os.path.exists(args.dataroot):
        os.mkdir(args.dataroot)

    main()
