import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

import argparse
from time import time

from utils import load_dataset, normalize_cifar, attack_pgd
from model import PreActResNet18

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True, type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    
    # avdersarial configerations
    parser.add_argument('--norm', default='l_inf', choices=['l_inf, l_2'])
    parser.add_argument('--eps', default=8., type=float)
    parser.add_argument('--alpha', default=2., type=float)
    parser.add_argument('--steps', default=10, type=int)

    # train configurations
    parser.add_argument('--ne', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--opt', default='SGD', choices=['SGD'])

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()



def lr_schedule(epoch):
    if epoch < args.ne * 0.75:
        return args.lr
    elif epoch < args.ne * 0.9:
        return args.lr * 0.1
    else:
        return args.lr * 0.01

if __name__ == '__main__':
    args = get_args()
    fname = args.fname
    if not os.path.exists(fname):
        os.mkdir(fname)
    
    with open(f'{fname}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    
    dataset = args.dataset
    device = f'cuda:{args.device}'
    model = PreActResNet18(10 if dataset == 'cifar10' else 100).to(device)
    train_loader, test_loader = load_dataset(dataset, args.bs)
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    eps = args.eps / 255.
    alpha = args.alpha / 255.
    steps = args.steps
    norm = args.norm

    best_acc = 0
    for epoch in range(args.ne):
        start_time = time()
        log_data = np.zeros(6) # train_clean, train_robust, train_cnt, test_clean, test_robust, test_cnt
        # train
        model.train()
        lr = lr_schedule(epoch)
        opt.param_groups[0].update(lr=lr)
        train_loader = tqdm(train_loader)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            delta = attack_pgd(model, x, y, eps, alpha, steps, norm)
                
            output = model(normalize_cifar(x + delta))
            loss = criterion(output, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            log_data[1] += (output.max(1)[1] == y).float().sum().item()
            log_data[2] += len(x)

            clean_output = model(normalize_cifar(x))
            log_data[0] += (clean_output.max(1)[1] == y).float().sum().item()

            train_loader.set_description(
                f'Epoch {epoch}: Train Loss {log_data[0]/log_data[2]:.4f} Acc {log_data[1]/log_data[2]*100:.2f}')
            if args.debug:
                break
        # test
        model.eval()
        test_loader = tqdm(test_loader)
        for x, y in test_loader:
            
            x, y = x.to(device), y.to(device)
            delta = attack_pgd(model, x, y, eps, alpha, steps, norm)

            clean_output = model(normalize_cifar(x))
            output = model(normalize_cifar(x+delta))
            #loss = criterion(output, y)
            
            log_data[3] += (clean_output.max(1)[1] == y).float().sum().item()
            log_data[4] += (output.max(1)[1] == y).float().sum().item()
            log_data[5] += len(x)

            train_loader.set_description(
                f'Epoch {epoch}: Test Loss {log_data[3]/log_data[5]:.4f} Acc {log_data[4]/log_data[5]*100:.2f}')
            if args.debug:
                break        
        if log_data[4] > best_acc:
            torch.save(model.state_dict(), f'{fname}/model.pth')
        

    
