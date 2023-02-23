import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

def load_dataset(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        transform_ = transforms.Compose([transforms.ToTensor()])
        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/kemove/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/kemove/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    
    elif dataset == 'cifar100':
        transform_ = transforms.Compose([transforms.ToTensor()])
        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/kemove/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/kemove/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

def attack_pgd(model, x, y, eps, alpha, n_iters, norm):
    delta = torch.zeros_like(x).to(x.device)
    if norm == "l_inf":
        delta.uniform_(-eps, eps)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*eps
    else:
        raise ValueError
    delta = torch.clamp(delta, 0-x, 1-x)
    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(normalize_cifar(x+delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        g = delta.grad.detach()
        if norm == "l_inf":
                d = torch.clamp(delta + alpha * torch.sign(g), min=-eps, max=eps).detach()
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=eps).view_as(d)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()
    
    return delta.detach()

