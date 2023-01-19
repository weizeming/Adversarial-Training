import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
from time import time

from utils import load_dataset, normalize_cifar
from model import PreActResNet18
from sam import SAM
from eval import attack_pgd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max-lr', default=0.1, type=float)
    parser.add_argument('--opt', default='SAM', choices=['SAM', 'SGD'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--rho', default=0.05, type=float)
    return parser.parse_args()

args = get_args()

def lr_schedule(epoch):
    if epoch < args.epochs * 0.75:
        return args.max_lr
    elif epoch < args.epochs * 0.9:
        return args.max_lr * 0.1
    else:
        return args.max_lr * 0.01

if __name__ == '__main__':
    
    dataset = args.dataset
    device = f'cuda:{args.device}'
    model = PreActResNet18(10 if dataset == 'cifar10' else 100).to(device)
    train_loader, test_loader = load_dataset(dataset, args.batch_size)
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()

    if args.opt == 'SGD': 
        opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'SAM':
        base_opt = torch.optim.SGD
        opt = SAM(params, base_opt,lr=args.max_lr, momentum=0.9, weight_decay=5e-4, rho=args.rho)
    
    all_log_data = []
    for epoch in range(args.epochs):
        start_time = time()
        log_data = [0,0,0,0] # train_loss, train_acc, test_loss, test_acc
        # train
        model.train()
        lr = lr_schedule(epoch)
        opt.param_groups[0].update(lr=lr)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if args.adv:  
                delta = attack_pgd(model, x, y, 8./255., 2./255, 5)
            else:
                delta = torch.zeros_like(x).to(x.device)
                
            output = model(normalize_cifar(x + delta))
            loss = criterion(output, y)
            
            if args.opt == 'SGD':
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            elif args.opt == 'SAM':
                loss.backward()
                opt.first_step(zero_grad=True)

                output_2 = model(normalize_cifar(x + delta))
                criterion(output_2, y).backward()
                opt.second_step(zero_grad=True)
            
            log_data[0] += (loss * len(y)).item()
            log_data[1] += (output.max(1)[1] == y).float().sum().item()
            
        # test
        model.eval()
        for x, y in test_loader:
            
            x, y = x.to(device), y.to(device)
            output = model(normalize_cifar(x))
            loss = criterion(output, y)
            
            log_data[2] += (loss * len(y)).item()
            log_data[3] += (output.max(1)[1] == y).float().sum().item()
        
        log_data = np.array(log_data)
        log_data[0] /= 60000
        log_data[1] /= 60000
        log_data[2] /= 10000
        log_data[3] /= 10000
        all_log_data.append(log_data)
        
        print(f'Epoch {epoch}:\t',log_data,f'\tTime {time()-start_time:.1f}s')
        torch.save(model.state_dict(), f'models/{dataset}_{args.opt}{"_adv" if args.adv else ""}.pth')
        
    all_log_data = np.stack(all_log_data,axis=0)
    
    df = pd.DataFrame(all_log_data)
    df.to_csv(f'logs/{dataset}_{args.opt}{"_adv" if args.adv else ""}.csv')
    
    
    
    plt.plot(all_log_data[:, [0,2]])
    plt.grid()
    plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Loss', fontsize=16)
    plt.legend(['train', 'test'], fontsize=16)
    plt.savefig(f'figs/{dataset}_{args.opt}{"_adv" if args.adv else ""}_loss.png', dpi=200)
    plt.clf()
    
    plt.plot(all_log_data[:, [1,3]])
    plt.grid()
    plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Acc', fontsize=16)
    plt.legend(['train', 'test'], fontsize=16)
    plt.savefig(f'figs/{dataset}_{args.opt}{"_adv" if args.adv else ""}_acc.png', dpi=200)
    plt.clf()
