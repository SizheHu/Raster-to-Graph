import argparse
import gc
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from args import get_args_parser
from datasets.dataset_demo import MyDataset_demo
from engine import train_one_epoch, evaluate_iter
from models.build import build_model, build_criterion, build_postprocessor
from util.output_utils import make_outputdir_and_log
from util.param_print_utils import match_name_keywords
from util.random_utils import set_random_seed
import numpy as np
import os, cv2, json
from tqdm import *
from PIL import Image
import math
import random
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from util.data_utils import data_to_cuda, delete_graphs, get_given_layers_random_region, get_random_region_targets, \
    draw_given_layers_on_tensors_random_region, initialize_tensors, nms, random_keep, is_stop, draw_preds_on_tensors, edge_inside, point_inside, \
    remove_points, remove_edge, get_edges_amount
from util.graph_utils import tensors_to_graphs_batch, get_cycle_basis_and_semantic, get_cycle_basis_and_semantic_2
from util.metric_utils import calculate_single_sample
from util.edges_utils import get_edges_alldirections
from util.misc import NestedTensor
import util.misc as utils
from util.visualize_utils import visualize_monte



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


dataset_test = MyDataset_demo('./data/dataset_v5/test')
# you can try num_workers>0 in linux
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
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



@torch.no_grad()
def demo_iter(model, criterion, postprocessor, data_loader, epoch, args):
    # make path
    Path(args.output_dir + '/val_visualize_iter' + '/epoch' + str(epoch)).mkdir(parents=True, exist_ok=True)

    model.eval()

    start_time = datetime.now()
    for batch_index, data in enumerate(data_loader):
        if 1:
            # data from cpu to cuda
            samples = data[0].to(torch.device('cuda:0'))
            targets = data[1]
            # masks
            masks = samples.decompose()[1]

            # all results
            monte_results = []
            tensors, unnormalized = initialize_tensors(samples.decompose()[0])
            monte_times = 1
            for _ in range(monte_times):
                tensors_this_monte = copy.deepcopy(tensors)
                preds = []
                # each iter
                for iter_time in range(9999999):
                    # node prediction
                    samples = NestedTensor(tensors_this_monte, masks)
                    outputs = model(samples)
                    target_sizes = torch.stack([torch.tensor([512, 512]).to(torch.device('cuda:0'))], dim=0)
                    results = postprocessor(outputs, target_sizes)[0]


                    this_preds = []
                    keep_confidence_threshold = 0.5
                    valid_label_indices_edges = torch.where(results['edges'] != 0)[0]
                    valid_label_indices_scores = torch.where(results['scores'] > keep_confidence_threshold)[0]
                    valid_label_indices = torch.tensor(
                        list(set(valid_label_indices_edges.tolist()).intersection(set(valid_label_indices_scores.tolist()))),
                        dtype=valid_label_indices_edges.dtype,
                        device=valid_label_indices_edges.device)
                    for valid_label_indice in valid_label_indices:
                        valid_results_i = {}
                        valid_results_i['scores'] = results['scores'][valid_label_indice]
                        valid_results_i['points'] = results['points'][valid_label_indice]
                        valid_results_i['last_edges'] = results['last_edges'][valid_label_indice]
                        valid_results_i['this_edges'] = results['this_edges'][valid_label_indice]
                        valid_results_i['edges'] = results['edges'][valid_label_indice]
                        valid_results_i['size'] = targets[0]['size']
                        valid_results_i['semantic_left_up'] = results['semantic_left_up'][valid_label_indice]
                        valid_results_i['semantic_right_up'] = results['semantic_right_up'][valid_label_indice]
                        valid_results_i['semantic_right_down'] = results['semantic_right_down'][valid_label_indice]
                        valid_results_i['semantic_left_down'] = results['semantic_left_down'][valid_label_indice]
                        this_preds.append(valid_results_i)
                    # NMS
                    # this_preds = nms(this_preds)
                    this_preds = random_keep(this_preds)

                    # terminate or not?
                    if is_stop(this_preds) == 1:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    elif is_stop(this_preds) == 2:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    else:
                        pass

                    # graph construction
                    # inter-level
                    last_edges = []
                    if len(preds) == 0:
                        pass
                    else:
                        all_given_points = []
                        for (given_points, given_last_edges, given_this_edges) in preds:
                            all_given_points.extend(given_points)
                        need_to_remove_in_last_edges = []
                        for this_pred in this_preds:
                            edge_search_threshold = 5
                            edge_search_bound_up = this_pred['points'].tolist()[1] - edge_search_threshold
                            edge_search_bound_left = this_pred['points'].tolist()[0] - edge_search_threshold
                            edge_search_bound_down = this_pred['points'].tolist()[1] + edge_search_threshold
                            edge_search_bound_right = this_pred['points'].tolist()[0] + edge_search_threshold

                            this_pred_last_edges = get_edges_alldirections(this_pred['last_edges'].item())
                            if int(this_pred_last_edges[0]):
                                upside_given_points = [p for p in all_given_points
                                                       if
                                                       (edge_search_bound_left <= p['points'].tolist()[0] <= edge_search_bound_right
                                                       and
                                                       p['points'].tolist()[1] < this_pred['points'].tolist()[1])]
                                if len(upside_given_points) == 0:
                                    need_to_remove_in_last_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for upside_given_point in upside_given_points:
                                        if this_pred['points'].tolist()[1] - upside_given_point['points'].tolist()[1] < mindist:
                                            nearest = upside_given_point
                                            mindist = this_pred['points'].tolist()[1] - upside_given_point['points'].tolist()[1]
                                    if edge_inside((nearest, this_pred), last_edges) or edge_inside((this_pred, nearest), last_edges):
                                        pass
                                    else:
                                        last_edges.append((this_pred, nearest))
                            if int(this_pred_last_edges[1]):
                                leftside_given_points = [p for p in all_given_points
                                                       if
                                                       (edge_search_bound_up <= p['points'].tolist()[1] <= edge_search_bound_down
                                                       and
                                                       p['points'].tolist()[0] < this_pred['points'].tolist()[0])]
                                if len(leftside_given_points) == 0:
                                    need_to_remove_in_last_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for leftside_given_point in leftside_given_points:
                                        if this_pred['points'].tolist()[0] - leftside_given_point['points'].tolist()[0] < mindist:
                                            nearest = leftside_given_point
                                            mindist = this_pred['points'].tolist()[0] - leftside_given_point['points'].tolist()[0]
                                    if edge_inside((nearest, this_pred), last_edges) or edge_inside((this_pred, nearest), last_edges):
                                        pass
                                    else:
                                        last_edges.append((this_pred, nearest))
                            if int(this_pred_last_edges[2]):
                                downside_given_points = [p for p in all_given_points
                                                       if
                                                       (edge_search_bound_left <= p['points'].tolist()[0] <= edge_search_bound_right
                                                       and
                                                       p['points'].tolist()[1] > this_pred['points'].tolist()[1])]
                                if len(downside_given_points) == 0:
                                    need_to_remove_in_last_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for downside_given_point in downside_given_points:
                                        if downside_given_point['points'].tolist()[1] - this_pred['points'].tolist()[1] < mindist:
                                            nearest = downside_given_point
                                            mindist = downside_given_point['points'].tolist()[1] - this_pred['points'].tolist()[1]
                                    if edge_inside((nearest, this_pred), last_edges) or edge_inside((this_pred, nearest), last_edges):
                                        pass
                                    else:
                                        last_edges.append((this_pred, nearest))
                            if int(this_pred_last_edges[3]):
                                rightside_given_points = [p for p in all_given_points
                                                       if
                                                       (edge_search_bound_up <= p['points'].tolist()[1] <= edge_search_bound_down
                                                       and
                                                       p['points'].tolist()[0] > this_pred['points'].tolist()[0])]
                                if len(rightside_given_points) == 0:
                                    need_to_remove_in_last_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for rightside_given_point in rightside_given_points:
                                        if rightside_given_point['points'].tolist()[0] - this_pred['points'].tolist()[0] < mindist:
                                            nearest = rightside_given_point
                                            mindist = rightside_given_point['points'].tolist()[0] - this_pred['points'].tolist()[0]
                                    if edge_inside((nearest, this_pred), last_edges) or edge_inside((this_pred, nearest), last_edges):
                                        pass
                                    else:
                                        last_edges.append((this_pred, nearest))

                        this_preds = remove_points(need_to_remove_in_last_edges, this_preds)
                        for (point1, point2) in copy.deepcopy(last_edges):
                            if point_inside(point1, need_to_remove_in_last_edges) or point_inside(point2, need_to_remove_in_last_edges):
                                last_edges = remove_edge((point1, point2), last_edges)

                    # terminate or not?
                    if is_stop(this_preds) == 1:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    elif is_stop(this_preds) == 2:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    else:
                        pass

                    # intra-level
                    this_edges = []
                    if len(this_preds) == 0 or len(this_preds) == 1:
                        pass
                    else:
                        need_to_remove_in_this_edges = []
                        for this_pred in this_preds:
                            edge_search_threshold = 5
                            edge_search_bound_up = this_pred['points'].tolist()[1] - edge_search_threshold
                            edge_search_bound_left = this_pred['points'].tolist()[0] - edge_search_threshold
                            edge_search_bound_down = this_pred['points'].tolist()[1] + edge_search_threshold
                            edge_search_bound_right = this_pred['points'].tolist()[0] + edge_search_threshold
                            this_pred_this_edges = get_edges_alldirections(this_pred['this_edges'].item())
                            if int(this_pred_this_edges[0]):
                                upside_this_points = [p for p in this_preds
                                                       if
                                                       (edge_search_bound_left <= p['points'].tolist()[0] <= edge_search_bound_right
                                                       and
                                                       p['points'].tolist()[1] < this_pred['points'].tolist()[1])]
                                if len(upside_this_points) == 0:
                                    need_to_remove_in_this_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for upside_this_point in upside_this_points:
                                        if this_pred['points'].tolist()[1] - upside_this_point['points'].tolist()[1] < mindist:
                                            nearest = upside_this_point
                                            mindist = this_pred['points'].tolist()[1] - upside_this_point['points'].tolist()[1]
                                    if int(get_edges_alldirections(nearest['this_edges'].item())[2]):
                                        if edge_inside((nearest, this_pred), this_edges) or edge_inside((this_pred, nearest), this_edges):
                                            pass
                                        else:
                                            this_edges.append((this_pred, nearest))
                                    else:
                                        need_to_remove_in_this_edges.append(this_pred)
                                        continue
                            if int(this_pred_this_edges[1]):
                                leftside_this_points = [p for p in this_preds
                                                         if
                                                         (edge_search_bound_up <= p['points'].tolist()[1] <= edge_search_bound_down
                                                          and
                                                          p['points'].tolist()[0] < this_pred['points'].tolist()[0])]
                                if len(leftside_this_points) == 0:
                                    need_to_remove_in_this_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for leftside_this_point in leftside_this_points:
                                        if this_pred['points'].tolist()[0] - leftside_this_point['points'].tolist()[0] < mindist:
                                            nearest = leftside_this_point
                                            mindist = this_pred['points'].tolist()[0] - leftside_this_point['points'].tolist()[0]
                                    if int(get_edges_alldirections(nearest['this_edges'].item())[3]):
                                        if edge_inside((nearest, this_pred), this_edges) or edge_inside((this_pred, nearest), this_edges):
                                            pass
                                        else:
                                            this_edges.append((this_pred, nearest))
                                    else:
                                        need_to_remove_in_this_edges.append(this_pred)
                                        continue
                            if int(this_pred_this_edges[2]):
                                downside_this_points = [p for p in this_preds
                                                       if
                                                       (edge_search_bound_left <= p['points'].tolist()[0] <= edge_search_bound_right
                                                       and
                                                       p['points'].tolist()[1] > this_pred['points'].tolist()[1])]
                                if len(downside_this_points) == 0:
                                    need_to_remove_in_this_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for downside_this_point in downside_this_points:
                                        if downside_this_point['points'].tolist()[1] - this_pred['points'].tolist()[1] < mindist:
                                            nearest = downside_this_point
                                            mindist = downside_this_point['points'].tolist()[1] - this_pred['points'].tolist()[1]
                                    if int(get_edges_alldirections(nearest['this_edges'].item())[0]):
                                        if edge_inside((nearest, this_pred), this_edges) or edge_inside((this_pred, nearest), this_edges):
                                            pass
                                        else:
                                            this_edges.append((this_pred, nearest))
                                    else:
                                        need_to_remove_in_this_edges.append(this_pred)
                                        continue
                            if int(this_pred_this_edges[3]):
                                rightside_this_points = [p for p in this_preds
                                                        if
                                                        (edge_search_bound_up <= p['points'].tolist()[1] <= edge_search_bound_down
                                                         and
                                                         p['points'].tolist()[0] > this_pred['points'].tolist()[0])]
                                if len(rightside_this_points) == 0:
                                    need_to_remove_in_this_edges.append(this_pred)
                                    continue
                                else:
                                    mindist = 99999
                                    nearest = None
                                    for rightside_this_point in rightside_this_points:
                                        if rightside_this_point['points'].tolist()[0] - this_pred['points'].tolist()[0] < mindist:
                                            nearest = rightside_this_point
                                            mindist = rightside_this_point['points'].tolist()[0] - this_pred['points'].tolist()[0]
                                    if int(get_edges_alldirections(nearest['this_edges'].item())[1]):
                                        if edge_inside((nearest, this_pred), this_edges) or edge_inside((this_pred, nearest), this_edges):
                                            pass
                                        else:
                                            this_edges.append((this_pred, nearest))
                                    else:
                                        need_to_remove_in_this_edges.append(this_pred)
                                        continue

                        this_preds = remove_points(need_to_remove_in_this_edges, this_preds)
                        for (point1, point2) in copy.deepcopy(last_edges):
                            if point_inside(point1, need_to_remove_in_this_edges) or point_inside(point2, need_to_remove_in_this_edges):
                                last_edges = remove_edge((point1, point2), last_edges)

                        for (point1, point2) in copy.deepcopy(this_edges):
                            if point_inside(point1, need_to_remove_in_this_edges) or point_inside(point2, need_to_remove_in_this_edges):
                                this_edges = remove_edge((point1, point2), this_edges)

                    # terminate or not?
                    if is_stop(this_preds) == 1:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    elif is_stop(this_preds) == 2:
                        edges_amount = get_edges_amount(preds)
                        monte_results.append((is_stop(this_preds), edges_amount, preds))
                        break
                    else:
                        pass
                    # this iter preds (G_t - G_t-1)
                    preds.append((this_preds, last_edges, this_edges))
                    # draw on image
                    tensors_this_monte, unnormalized_this_monte = draw_preds_on_tensors(preds, tensors_this_monte)

            # best result
            sorted_results = sorted([tupl for tupl in monte_results if tupl[0] == 2], key=lambda x: x[1], reverse=True)
            sorted_results_incomplete = sorted([tupl for tupl in monte_results if tupl[0] == 1], key=lambda x: x[1], reverse=True)
            if len(sorted_results) > 0:
                best_result = sorted_results[0]
            else:
                best_result = sorted_results_incomplete[0]


            # room extraction
            d_rev, simple_cycles, results = get_cycle_basis_and_semantic(best_result)
            # print(d_rev, simple_cycles, results)

            # visualize
            if 1:
                visualize_monte(unnormalized, best_result, epoch, args.output_dir, batch_index + 1, d_rev, simple_cycles, results)

        
            # print metrics
            count = batch_index + 1
            if count % 1 == 0:
                print('Epoch:', epoch, '\t',
                      str(count) + '/' + str(len(data_loader))
                      )

    end_time = datetime.now()

    # print metrics
    print('demo time:', time.strftime("%H:%M:%S", time.gmtime((end_time - start_time).seconds)))

demo_iter(model, criterion, postprocessor, data_loader_test, start_epoch, args)