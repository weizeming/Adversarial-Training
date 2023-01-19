import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
from torchvision import datasets, transforms

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