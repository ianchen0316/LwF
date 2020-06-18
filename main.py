""" Main Script For Incremental Learning """

import argparse
import random
import pickle
import logging
import copy
import os
import sys
sys.path.append('../../semi-supervised/fixmatch')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, datasets

import numpy as np

from datagen import BatchScenario, LabelTransformed, UnlabelTransformed, EvalTransformed
from model import ModelSetup
from utility import evaluate, save_checkpoint

from train_lwf import supervised_LwF, DistillLoss

from randaugment import RandAugmentMC
from model import ModelSetup
from train import get_cosine_schedule_with_warmup, FixmatchLoss, fixmatch_train

#TODO: 
# - Modify test_base to make it consistent with train_base dataset
# Add load from state dict mechanism

# ============ Parse Arguments ================
parser = argparse.ArgumentParser(description='LwF settings')

parser.add_argument('--base_path', default='./states/base_best.pth.tar', type=str, help='path for base checkpoint')
parser.add_argument('--scenario', default='supervised', type=str, help='training scenario. Can be supervised or semi-supervised')
parser.add_argument('--exp_name', default='LwF-supervised', type=str, help='name of experiment')

parser.add_argument('--model_name', default='WideResnet', type=str, help='backbone model for classification')
parser.add_argument('--initial_class', default=5, type=int, help='initial class in old model')
parser.add_argument('--increment_class', default=5, type=int, help='Number of new classes for each class expansion')

parser.add_argument('--lr', default=0.03, type=float, help='Initial learning rate')
parser.add_argument('--l_u', type=float, default=1.0, help='weight of unlabeled loss')
parser.add_argument('--threshold', type=int, default=0.95, help='probability threshold for pseudo label')
parser.add_argument('--finetune_epochs', default=10, type=int, help='Number of initial finetuning epochs')
parser.add_argument('--epochs', default=1024, type=int, help='Number of epochs')
parser.add_argument('--aug_num', default=2**16, type=int, help='number of augmented labeled data')
parser.add_argument('--batch_size', default=64, type=int, help='Size for each batch')
parser.add_argument('--weight_decay', default=0.001, type=float, help='coefficient of L2 regularization loss term')
parser.add_argument('--momentum', default=0.5, type=float, help='momentum of optimizer')
parser.add_argument('--temperature', default=2.0, type=float, help='Logit temperature')

parser.add_argument('--seed', type=int, default=42, help='seed for randomization')

parser.add_argument('--state_path', default='./states', type=str, help='path for states')
parser.add_argument('--result_path', default='./history', type=str, help='path for results')

args = parser.parse_args()

# ========== Set Loggers and Result Trackers ===================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", 
    level=logging.INFO
)

os.makedirs(args.state_path, exist_ok=True)
os.makedirs(args.result_path, exist_ok=True)

results = {}
results['train_loss'] = []
results['train_loss_d'] = []
results['train_loss_c'] = []
results['test_acc'] = []
results['per_class_test_acc'] = []
results['test_loss'] = []

# ======== Set Seeds ===========================================================

if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


# =================== Define Transformations ==================================

weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ])


strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ])
    

eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))   
        ])

# =================== Scenario Generation =======================================
config_map = {
    'D_0': [(5000, 0), (5000, 0), (5000, 0), (5000, 0), (5000, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    'D_1': [ (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (5000, 0), (5000, 0), (5000, 0), (5000, 0), (5000, 0)]  
}

scenario = BatchScenario('cifar-10', './', config_map)
scenario.scenario_generation()
train_base, labeled_ind, unlabeled_ind = scenario.get_batch_dataset('D_1')
test_dataset = scenario.get_test_dataset()
    
labeled = LabelTransformed(train_base, labeled_ind, args, weak_transform)
if args.scenario == 'semi-supervised':
    unlabeled = UnlabelTransformed(train_base, unlabeled_ind, args, weak_transform, strong_transform)
test = EvalTransformed(test_dataset, eval_transform)
    
labeled_iterator = DataLoader(labeled, sampler=RandomSampler(labeled), batch_size=args.batch_size, drop_last=True)
if args.scenario == 'semi-supervised':
    unlabeled_iterator = DataLoader(unlabeled, sampler=RandomSampler(unlabeled), batch_size=args.mu*args.batch_size, drop_last=True)
test_iterator = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    
args.num_iters = args.aug_num // args.batch_size
    
# =============== Device Setting =================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============== Tracing History Setting =======================
logger.info(dict(args._get_kwargs()))

# =============== Modeling ======================================

# Old Model
model_setup = ModelSetup(args.initial_class, args.model_name)
old_model = model_setup.get_model()

previous_state = torch.load(args.base_path)
old_model.load_state_dict(previous_state['state_dict'])

for param in old_model.parameters():
    param.requires_grad = False
    
# New Model
model = copy.deepcopy(old_model)

old_weight = model.classifier.weight
old_bias = model.classifier.bias

model.classifier.weight = torch.nn.Parameter((torch.cat((old_weight, torch.randn(size=(args.increment_class, 64))))))
model.classifier.bias = torch.nn.Parameter((torch.cat((old_bias, torch.randn(size=(args.increment_class, ))))))

# To Device
old_model.to(device)
model.to(device)

logger.info("Old Model: Total Parameters: {}M".format(sum(p.numel() for p in old_model.parameters()) / 1e6))
logger.info("New Model: Total Parameters: {}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

# ===================== Loss Function / Optimizer ==============================

if args.scenario == 'supervised':
    distill_loss = DistillLoss()
    classification_loss = nn.CrossEntropyLoss()
elif args.scenario == 'semi-supervised': 
    pass
    # loss_func = HybridLossWithUnlabeled()
test_loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs*args.num_iters)

# ===================== Incremental Learning Training =========================

start_epoch = 0
best_acc = 0

for epoch in range(args.epochs):

    loss, loss_d, loss_c = supervised_LwF(epoch, model, old_model, labeled_iterator, distill_loss, classification_loss, args, optimizer, lr_scheduler, device)
    test_acc, per_class_ters_acc, test_loss = evaluate(model, test_iterator, 10, test_loss_func, device)
    
    results['train_loss'].append(loss)
    results['train_loss_d'].append(loss_d)
    results['train_loss_c'].append(loss_c)
    results['test_acc'].append(test_acc)
    results['per_class_test_acc'].append(per_class_ters_acc)
    results['test_loss'].append(test_loss)
    
    with open(args.result_path + '/' + args.exp_name + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save state for each epoch
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    
    save_checkpoint(state, is_best, args.state_path, args.exp_name)
    
    # Log for test accuracy
    logger.info('Best Accuracy: {}'.format(best_acc))
    logger.info('Epoch {} | Test Accuracy: {}'.format(epoch+1, test_acc))