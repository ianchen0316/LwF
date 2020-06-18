import os
import shutil

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

import numpy 


def save_checkpoint(state, is_best, ck_path, exp_name):
    
    """ 
    Checkpoint Mechanism for Training. Save current state for each epoch at ck_path for tracking.
    If the performance is currently the best, then copy the state file to best_path for future model usage.

    """

    history_path = os.path.join(ck_path, exp_name + '_checkpoint.pth.tar')
    torch.save(state, history_path)
    
    if is_best:
        shutil.copyfile(history_path, os.path.join(ck_path, exp_name + '_best.pth.tar'))
        

def evaluate(model, test_iterator, num_class, loss_func ,device):
    
    model.eval()
    loss_func.to(device)
    
    epoch_loss = 0.0
    
    correct_counts = torch.zeros(num_class)
    sample_counts = torch.zeros(num_class)
    
    with torch.no_grad():
        
        for (x_batch, y_batch) in test_iterator:
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_hat = model(x_batch)
            
            batch_correct_counts, batch_sample_counts = per_class_count(y_hat, y_batch, num_class)
            batch_loss = loss_func(y_hat, y_batch)
            
            correct_counts += batch_correct_counts
            sample_counts += batch_sample_counts
            
            epoch_loss += batch_loss.item()
    
    acc = correct_counts / sample_counts

    return torch.mean(acc), acc, epoch_loss / len(test_iterator)


def per_class_count(y_hat, y, num_class):
    
    correct_counts = []
    sample_counts = []
    
    predictions = torch.argmax(y_hat, 1)
    
    for i in range(num_class):
        num_correct = ((predictions == y) & (y == i)).sum()
        num_sample = (y == i).sum()
        correct_counts.append(num_correct)
        sample_counts.append(num_sample)
  
    return torch.Tensor(correct_counts), torch.Tensor(sample_counts)


def calculate_accuracy(y_hat, y):
    
    predictions = torch.argmax(y_hat, 1)
    num_correct = (predictions == y).sum()
    acc = num_correct.float() / y_hat.size()[0]
    
    return acc