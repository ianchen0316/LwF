""" Scenario Generation of Lifelong Learning """

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class BatchScenario:
    
    def __init__(self, dataset, root, config_map):

        if dataset == 'cifar-10':
        
            self.train_base = datasets.CIFAR10(root, train=True, download=True)
            
            self._train_data = self.train_base.data
            self._train_labels = np.array(self.train_base.targets)
            self._class_indices = [np.where(self._train_labels == i)[0] for i in range(10)]
            
            self.config_map = config_map
            self.total_batch_indices = None 
            
            test_base = datasets.CIFAR10(root, train=False, download=True)
            
            self._test_data = test_base.data
            self._test_labels = np.array(test_base.targets)
            self._test_class_indices = [np.where(self._test_labels == i)[0] for i in range(10)]
            
    def scenario_generation(self):
        
        batch_names = list(self.config_map.keys())
        num_classes = len(self.config_map[batch_names[0]])
        
        total_batches = {}
        for name in batch_names:
            total_batches[name] = {'labeled_ind': [], 'unlabeled_ind': []}
            
        for i in range(num_classes):
            cursor = 0
            np.random.shuffle(self._class_indices[i])
            for name in batch_names:
                n_per_class_labeled = self.config_map[name][i][0]
                total_batches[name]['labeled_ind'].extend(self._class_indices[i][cursor:cursor+n_per_class_labeled])
                cursor += n_per_class_labeled
                # print(self.total_batch_indices['batch_0']['labeled_ind'])
                n_per_class_unlabeled = self.config_map[name][i][1]
                total_batches[name]['unlabeled_ind'].extend(self._class_indices[i][cursor:cursor+n_per_class_unlabeled])
                cursor += n_per_class_unlabeled
        
        self.total_batch_indices = total_batches
        
    def get_batch_dataset(self, batch_name):
            
        labeled_ind = self.total_batch_indices[batch_name]['labeled_ind']
        unlabeled_ind = self.total_batch_indices[batch_name]['unlabeled_ind']
      
        return self.train_base, labeled_ind, unlabeled_ind
    
    def get_test_dataset(self, selected_class=None):
        
        if not selected_class:
            test_dataset = (self._test_data, self._test_labels)
            return test_dataset
        selected_indices = []
        for c in selected_class:
            selected_indices.extend(self._test_class_indices[c])
        test_dataset = (self._test_data[selected_indices], self._test_labels[selected_indices])
        
        return test_dataset
        
        
class LabelTransformed(Dataset):
    
    def __init__(self, train_base, labeled_ind, args, transform):
        
        aug_ratio = args.aug_num // len(labeled_ind) 
        
        self.train_base = train_base
        self.aug_labeled_ind = np.hstack([labeled_ind for _ in range(aug_ratio)])
        
        if len(self.aug_labeled_ind) < args.aug_num:
            diff = args.aug_num - len(self.aug_labeled_ind)
            self.aug_labeled_ind = np.hstack((self.aug_labeled_ind, np.random.choice(self.aug_labeled_ind, diff)))
        
        self.transform = transform
    
    def __len__(self):
        
        return len(self.aug_labeled_ind)
    
    def __getitem__(self, ind):
    
        (img, label) = self.train_base.data[self.aug_labeled_ind[ind]], np.array(self.train_base.targets)[self.aug_labeled_ind[ind]]
        img = self.transform(img)
        
        return img, label
       
            
class UnlabelTransformed(Dataset):
    
    def __init__(self, train_base, unlabeled_ind, args, weak_transform, strong_transform):
        
        aug_ratio = args.mu*args.aug_num // len(unlabeled_ind) 
        
        self.train_base = train_base
        self.aug_unlabeled_ind = np.hstack([unlabeled_ind for _ in range(aug_ratio)])
        
        if len(self.aug_unlabeled_ind) < args.mu*args.aug_num:
            diff = args.mu*args.aug_num - len(self.aug_unlabeled_ind)
            self.aug_unlabeled_ind = np.hstack((self.aug_unlabeled_ind, np.random.choice(self.aug_unlabeled_ind, diff)))
        
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        
        return len(self.aug_unlabeled_ind)
    
    def __getitem__(self, ind):
    
        img = self.train_base.data[self.aug_unlabeled_ind[ind]]
        img_weak = self.weak_transform(img)
        img_strong = self.strong_transform(img)
        
        return img_weak, img_strong                
            

class EvalTransformed(Dataset):
    
    def __init__(self, eval_dataset, transform):
        
        self.eval_dataset = eval_dataset[0]
        self.eval_target = eval_dataset[1]
        self.transform = transform
    
    def __len__(self):
        
        return len(self.eval_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.eval_dataset[ind], self.eval_target[ind]
        img = self.transform(img)
        
        return img, label