import argparse
import gc
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from args import get_args_parser
from datasets.dataset import MyDataset
from engine import train_one_epoch, evaluate_iter
from models.build import build_model, build_criterion, build_postprocessor
from util.output_utils import make_outputdir_and_log
from util.param_print_utils import match_name_keywords
from util.random_utils import set_random_seed
import numpy as np

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)
gc.collect()
torch.cuda.empty_cache()
args = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args()
make_outputdir_and_log(args)

device = torch.device(args.device)
set_random_seed(args)
model = build_model(args)
criterion = build_criterion(args)
postprocessor = build_postprocessor()

dataset_train = MyDataset(args.dataset_path + '/train', args.dataset_path + '/annot_json' + '/instances_train.json', extract_roi=False)
dataset_val = MyDataset(args.dataset_path + '/val', args.dataset_path + '/annot_json' + '/instances_val.json', extract_roi=False)

# you can try num_workers>0 in linux
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                               drop_last=True,
                               collate_fn=utils.collate_fn, num_workers=0,
                               pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                             drop_last=False, collate_fn=utils.collate_fn, num_workers=0,
                             pin_memory=True)

param_dicts = [
    {
        "params":
            [p for n, p in model.named_parameters()
             if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
        "lr": args.lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if
                   match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
        "lr": args.lr_backbone,
    },
    {
        "params": [p for n, p in model.named_parameters() if
                   match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
        "lr": args.lr * args.lr_linear_proj_mult,
    }
]
for n, p in model.named_parameters():
    print(n)

if args.optim == 'SGD':
    optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=0.1)

start_epoch = 0
max_epoch = 800

# training from checkpoint
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for ind, pg in enumerate(optimizer.param_groups):
        if ind == 0:
            pg['lr'] = args.lr
        elif ind == 1:
            pg['lr'] = args.lr_backbone
        elif ind == 2:
            pg['lr'] = args.lr * args.lr_linear_proj_mult
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1

# training & evaluating
for epoch in range(start_epoch, max_epoch):
    # train 1 epoch
    train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args.clip_max_norm, args)
    lr_scheduler.step()
    # save a checkpoint
    if args.output_dir:
        checkpoint_path = args.output_dir + '/' + f'checkpoint{epoch:04}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)
    # evaluate
    evaluate_iter(model, criterion, postprocessor, data_loader_val, epoch, args)