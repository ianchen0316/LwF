import sys
sys.path.append('../../semi-supervised/')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility import calculate_accuracy

from fixmatch.train import FixmatchLoss, fixmatch_train


def supervised_train(model, iterator, loss_func, optimizer, lr_scheduler, device):
    
    model.train()
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    for (x_batch, y_batch) in iterator:
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_hat = model(x_batch)
        
        batch_loss = loss_func(y_hat, y_batch)
        batch_acc = calculate_accuracy(y_hat, y_batch)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def supervised_LwF(epoch, model, old_model, labeled_iterator, distill_loss, classifcation_loss, args, optimizer, lr_scheduler, device):
    
    model.train()
    
    for (X_batch, Y_batch) in labeled_iterator:
        
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        Y_hat = model(X_batch)
        
        with torch.no_grad():
            Yo = old_model(X_batch)
        
        # Loss function of LwF
        loss_d = distill_loss(Y_hat, Yo, args)    
        loss_c = classifcation_loss(Y_hat, Y_batch)
        loss = loss_d + loss_c
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
    return loss.item(), loss_d.item(), loss_c.item()


def semi_LwF(epoch, model, old_model, labeled_iterator, unlabeled_iterator, loss_func, num_iters, threshold, optimizer, lr_scheduler, device):
    
    model.train()
    
    l_iterator, u_iterator = iter(labeled_iterator), iter(unlabeled_iterator)
    
    for i in range(num_iters):
        
        try:
            X_weak, Y_target = next(l_iterator)
            X_weak, Y_target = X_weak.to(device), Y_target.to(device)
        except:
            l_iterator = iter(labeled_iterator)
            X_weak, Y_target = next(l_iterator)
            X_weak, Y_target = X_weak.to(device), Y_target.to(device)
        try:
            U_weak, U_strong = next(u_iterator)
            U_weak, U_strong = U_weak.to(device), U_strong.to(device)
        except:
            u_iterator = iter(unlabeled_iterator)
            U_weak, U_strong = next(u_iterator)
            U_weak, U_strong = U_weak.to(device), U_strong.to(device)

        Y_target = Y_target.to(device)
    
        with torch.no_grad():
            Yo = old_model(X_weak).to(device)
            Yo_u_weak = old_model(U_weak).to(device)
    
        # ==========================
        
        l_batch_size = X_weak.size(0)
        u_batch_size = U_weak.size(0)

        total_imgs = torch.cat([X_weak, U_weak, U_strong], dim=0).to(device)
        logits = model(total_imgs)
    
        # ==========================
        logits_x = logits[:l_batch_size]
        logits_u_weak, logits_u_strong = logits[l_batch_size:l_batch_size+u_batch_size], logits[l_batch_size+u_batch_size:]
    
        # Compute Pseudo-Label of u_weak:
        with torch.no_grad():
            probs = torch.softmax(logits_u_weak, dim=1)
            max_p, guess_labels = torch.max(probs, dim=1)
            mask = max_p.ge(threshold).float()
        
        loss, loss_x, loss_u, loss_o = loss_func(logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, Yo, device)    
        #loss, loss_x, loss_u, loss_o = loss_func(logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, Yo, Yo_u_weak, device)  
    
        # ============================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
    return model, loss_x, loss_u, loss_o


class DistillLoss:
    
    def __call__(self, logits, old_logits, args):
        
        T = args.temperature
        old_prob = F.softmax(old_logits / T)
        new_output = F.log_softmax(logits[:, :args.initial_class] / T)
        loss_d = -torch.mean(torch.sum(new_output*old_prob, dim=1, keepdim=False), dim=0, keepdim=False)
        
        return loss_d

    
class HybridLoss:

    def __call__(self, logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, Yo, device):

        lambda_u = 1
        lambda_o = 10
        
        # Calculate the loss of labeled batch data 
        l_loss_func = nn.CrossEntropyLoss().to(device)
        loss_x = l_loss_func(logits_x, Y_target).to(device)
        
        # Calculate the loss of unlabeled batch data
        u_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
        loss_u  = (u_loss_func(logits_u_strong, guess_labels)*mask).mean().to(device)

        # Calculate the loss between old task outputs while model evolves
        T = 2
        old_prob = F.softmax(Yo / T)
        new_output = F.log_softmax(logits_x[:, :5] / T)
        loss_o = -torch.mean(torch.sum(new_output*old_prob, dim=1, keepdim=False), dim=0, keepdim=False)
        #old_prob = Yo / T
        #new_output = logits_x[:, :5] / T
        #loss_o = torch.mean(torch.sum((old_prob - new_output)**2, dim=1, keepdim=False), dim=0, keepdim=False)
        
        loss = loss_x + lambda_u*loss_u + lambda_o*loss_o

        return loss, loss_x, loss_u, loss_o
    
    
class HybridLossWithUnlabeled:

    def __call__(self, logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, Yo, Yo_u_weak, device):

        lambda_u = 1
        lambda_o = 10
        
        # Calculate the loss of labeled batch data 
        l_loss_func = nn.CrossEntropyLoss().to(device)
        loss_x = l_loss_func(logits_x, Y_target).to(device)
        
        # Calculate the loss of unlabeled batch data
        u_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
        loss_u  = (u_loss_func(logits_u_strong, guess_labels)*mask).mean().to(device)

        # Calculate the loss between old task outputs while model evolves
        T = 2
        old_prob = F.softmax(Yo / T)
        new_output = F.log_softmax(logits_x[:, :5] / T)
        
        old_prob_u = F.softmax(Yo_u_weak / T)
        new_output_u = F.log_softmax(logits_u_weak[:, :5] / T)
        
        loss_o = -torch.mean(torch.sum(torch.cat((new_output*old_prob, new_output_u*old_prob_u)), dim=1, keepdim=False), dim=0, keepdim=False)
        #old_prob = Yo / T
        #new_output = logits_x[:, :5] / T
        #loss_o = torch.mean(torch.sum((old_prob - new_output)**2, dim=1, keepdim=False), dim=0, keepdim=False)
        
        loss = loss_x + lambda_u*loss_u + lambda_o*loss_o

        return loss, loss_x, loss_u, loss_o