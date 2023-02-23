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


DATA_SHAPE = 32
def init_sampler(loader):
    im_test, targ_test = [], []
    for _, (im, targ) in enumerate(loader):
        im_test.append(im)
        targ_test.append(targ)
    im_test, targ_test = torch.cat(im_test), torch.cat(targ_test)

    conditionals = []
    for i in range(10):
        imc = im_test[targ_test == i]
        down_flat = downsample(imc).view(len(imc), -1)
        mean = down_flat.mean(dim=0)
        down_flat = down_flat - mean.unsqueeze(dim=0)
        cov = down_flat.t() @ down_flat / len(imc)
        dist = MultivariateNormal(mean, covariance_matrix=cov+1e-4*torch.eye(3 * DATA_SHAPE//1 * DATA_SHAPE//1))
        conditionals.append(dist)

    return conditionals

def init_seed(sampler, num_example):
    
    img_seed = []

    for i in range(10):
        for _ in range(num_example):
            img_seed.append(sampler[i].sample().view(3, DATA_SHAPE//1, DATA_SHAPE//1))
    img_seed = torch.stack(img_seed)
    img_seed = upsample(torch.clamp(img_seed, min=0, max=1))
    return img_seed

def downsample(x, step=1):
    down = torch.zeros([len(x), 3, DATA_SHAPE//step, DATA_SHAPE//step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down

def upsample(x, step=1):
    up = torch.zeros([len(x), 3, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up
