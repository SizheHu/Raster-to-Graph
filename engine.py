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


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, max_norm, args):
    start_time = datetime.now()

    # set to train mode
    model.train()
    criterion.train()

    # total loss
    total_loss_value = 0
    total_loss_point_value = 0
    total_loss_edge_value = 0
    total_loss_last_edge_value = 0
    total_loss_this_edge_value = 0
    total_loss_semantic_left_up_value = 0
    total_loss_semantic_right_up_value = 0
    total_loss_semantic_right_down_value = 0
    total_loss_semantic_left_down_value = 0

    for batch_index, data in enumerate(data_loader):
        # data
        samples, targets = data_to_cuda(data[0], data[1])
        graphs = tensors_to_graphs_batch([t['graph'] for t in targets])
        targets = delete_graphs(targets)

        # get randomized inputs and targets
        given_layers = get_given_layers_random_region(targets, graphs)
        random_layer_targets = get_random_region_targets(given_layers, graphs, targets)
        tensors = draw_given_layers_on_tensors_random_region(given_layers, samples.decompose()[0], graphs)[0]
        masks = samples.decompose()[1]
        samples = NestedTensor(tensors, masks)

        optimizer.zero_grad()
        # model output
        outputs = model(samples)

        # all losses
        loss_dict = criterion(outputs, random_layer_targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_weighted = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}

        loss_value = losses.item()
        total_loss_value += loss_value

        loss_point_value = loss_dict_weighted['loss_point'].item()
        total_loss_point_value += loss_point_value

        loss_edge_value = loss_dict_weighted['loss_edge'].item()
        total_loss_edge_value += loss_edge_value

        loss_last_edge_value = loss_dict_weighted['loss_last_edge'].item()
        total_loss_last_edge_value += loss_last_edge_value

        loss_this_edge_value = loss_dict_weighted['loss_this_edge'].item()
        total_loss_this_edge_value += loss_this_edge_value

        loss_semantic_left_up_value = loss_dict_weighted['loss_semantic_left_up'].item()
        total_loss_semantic_left_up_value += loss_semantic_left_up_value

        loss_semantic_right_up_value = loss_dict_weighted['loss_semantic_right_up'].item()
        total_loss_semantic_right_up_value += loss_semantic_right_up_value

        loss_semantic_right_down_value = loss_dict_weighted['loss_semantic_right_down'].item()
        total_loss_semantic_right_down_value += loss_semantic_right_down_value

        loss_semantic_left_down_value = loss_dict_weighted['loss_semantic_left_down'].item()
        total_loss_semantic_left_down_value += loss_semantic_left_down_value

        # print loss
        count = batch_index + 1
        if count % 10 == 1:
            print('Epoch:', epoch, '\t',
                  str(count) + '/' + str(len(data_loader)), '\t',
                  'lr:', format(optimizer.param_groups[0]["lr"], '.6f'), '\t',
                  'loss:', format(loss_value, '.4f'), '\t',
                  'loss_edge:', format(loss_edge_value, '.4f'), '\t',
                  'loss_last_edge:', format(loss_last_edge_value, '.4f'), '\t',
                  'loss_this_edge:', format(loss_this_edge_value, '.4f'), '\t',
                  'loss_semantic_left_up:', format(loss_semantic_left_up_value, '.4f'), '\t',
                  'loss_semantic_right_up:', format(loss_semantic_right_up_value, '.4f'), '\t',
                  'loss_semantic_right_down:', format(loss_semantic_right_down_value, '.4f'), '\t',
                  'loss_semantic_left_down:', format(loss_semantic_left_down_value, '.4f'), '\t',
                  'loss_point:', format(loss_point_value, '.4f'), '\t')

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    # print avg loss in the epoch
    print('avg_loss:', format(total_loss_value / len(data_loader), '.4f'), '\t',
          'avg_loss_edge:', format(total_loss_edge_value / len(data_loader), '.4f'), '\t',
          'avg_loss_last_edge:', format(total_loss_last_edge_value / len(data_loader), '.4f'), '\t',
          'avg_loss_this_edge:', format(total_loss_this_edge_value / len(data_loader), '.4f'), '\t',

          'avg_loss_semantic_left_up:', format(total_loss_semantic_left_up_value / len(data_loader), '.4f'), '\t',
          'avg_loss_semantic_right_up:', format(total_loss_semantic_right_up_value / len(data_loader), '.4f'), '\t',
          'avg_loss_semantic_right_down:', format(total_loss_semantic_right_down_value / len(data_loader), '.4f'), '\t',
          'avg_loss_semantic_left_down:', format(total_loss_semantic_left_down_value / len(data_loader), '.4f'), '\t',

          'avg_loss_point:', format(total_loss_point_value / len(data_loader), '.4f'))
    end_time = datetime.now()
    print('epoch train_time:', time.strftime("%H:%M:%S", time.gmtime((end_time - start_time).seconds)))

    # log
    f = open(args.output_dir + '/train_log.txt', mode='a')
    f_str = 'Epoch:' + str(epoch) + '\t' + 'lr:' + str(format(optimizer.param_groups[0]["lr"], '.6f')) + '\t' + \
            'avg_loss:' + str(format(total_loss_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_edge:' + str(format(total_loss_edge_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_last_edge:' + str(format(total_loss_last_edge_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_this_edge:' + str(format(total_loss_this_edge_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_semantic_left_up:' + str(format(total_loss_semantic_left_up_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_semantic_right_up:' + str(format(total_loss_semantic_right_up_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_semantic_right_down:' + str(format(total_loss_semantic_right_down_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_semantic_left_down:' + str(format(total_loss_semantic_left_down_value / len(data_loader), '.4f')) + '\t' + \
            'avg_loss_point:' + str(format(total_loss_point_value / len(data_loader), '.4f')) + '\t' + \
            'epoch train_time:' + time.strftime("%H:%M:%S", time.gmtime((end_time - start_time).seconds)) + '\n'
    f.write(f_str)
    f.close()


@torch.no_grad()
def evaluate_iter(model, criterion, postprocessor, data_loader, epoch, args):
    # make path
    Path(args.output_dir + '/val_visualize_iter' + '/epoch' + str(epoch)).mkdir(parents=True, exist_ok=True)

    model.eval()
    criterion.eval()

    # metrics
    all_points_TP = 0
    all_points_FP = 0
    all_points_FN = 0
    all_edges_TP = 0
    all_edges_FP = 0
    all_edges_FN = 0
    all_regions_TP = 0
    all_regions_FP = 0
    all_regions_FN = 0
    all_rooms_TP = 0
    all_rooms_FP = 0
    all_rooms_FN = 0
    # metrics of "Structure"
    perfect_result_graph = 0
    # metrics of "Overall"
    perfect_result_plan = 0

    # all results
    all_dict = {}
  
    start_time = datetime.now()
    for batch_index, data in enumerate(data_loader):
        if 1:
            # data from cpu to cuda
            samples, targets = data_to_cuda(data[0], data[1])
            # masks
            masks = samples.decompose()[1]
            # targets
            graphs = tensors_to_graphs_batch([t['graph'] for t in targets])
            targets = delete_graphs(targets)

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
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
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
            # d_rev, simple_cycles, results = get_cycle_basis_and_semantic_2(best_result)

            # visualize
            if 0:
                visualize_monte(unnormalized, best_result, epoch, args.output_dir, batch_index + 1, d_rev, simple_cycles, results)

            # gt
            tgt_this_preds = []
            tgt_this_edges = []
            for _ in range(len(targets[0]['points'])):
                tgt_p_d = {}
                tgt_p_d['scores'] = torch.tensor(1.0000, device='cuda:0')
                tgt_p_d['points'] = targets[0]['unnormalized_points'][_]
                tgt_p_d['edges'] = targets[0]['edges'][_]
                tgt_p_d['size'] = targets[0]['size']
                tgt_p_d['semantic_left_up'] = targets[0]['semantic_left_up'][_]
                tgt_p_d['semantic_right_up'] = targets[0]['semantic_right_up'][_]
                tgt_p_d['semantic_right_down'] = targets[0]['semantic_right_down'][_]
                tgt_p_d['semantic_left_down'] = targets[0]['semantic_left_down'][_]
                tgt_this_preds.append(tgt_p_d)
                for __ in range(4):
                    adj = graphs[0][tuple(tgt_p_d['points'].tolist())][__]
                    if adj != (-1, -1):
                        tgt_p_d1 = tgt_p_d
                        tgt_p_d2 = {}
                        indx = 99999
                        for ___, up in enumerate(targets[0]['unnormalized_points'].tolist()):
                            if abs(up[0] - adj[0]) + abs(up[1] - adj[1]) <= 2:
                                indx = ___
                                break
                        assert indx != 99999
                        # tgt_p_d2['scores'] = torch.tensor(1.0000, device='cuda:0')
                        tgt_p_d2['points'] = targets[0]['unnormalized_points'][indx]
                        tgt_p_d2['edges'] = targets[0]['edges'][indx]
                        tgt_p_d2['size'] = targets[0]['size']
                        tgt_p_d2['semantic_left_up'] = targets[0]['semantic_left_up'][indx]
                        tgt_p_d2['semantic_right_up'] = targets[0]['semantic_right_up'][indx]
                        tgt_p_d2['semantic_right_down'] = targets[0]['semantic_right_down'][indx]
                        tgt_p_d2['semantic_left_down'] = targets[0]['semantic_left_down'][indx]
                        tgt_e_l = (tgt_p_d1, tgt_p_d2)
                        if not edge_inside((tgt_p_d2, tgt_p_d1), tgt_this_edges):
                            tgt_this_edges.append(tgt_e_l)
            tgt = [(tgt_this_preds, [], tgt_this_edges)]

            target_d_rev, target_simple_cycles, target_results = \
                get_cycle_basis_and_semantic((2, 999999, tgt))
            points_TP, points_FP, points_FN, edges_TP, edges_FP, edges_FN, dist_error,\
                regions_TP, regions_FP, regions_FN, rooms_TP, rooms_FP, rooms_FN = \
                calculate_single_sample(best_result, graphs[0],
                                        target_d_rev, target_simple_cycles, target_results,
                                        d_rev, simple_cycles, results)

            all_dict[batch_index + 1] = [points_TP, points_FP, points_FN, edges_TP, edges_FP, edges_FN, regions_TP, regions_FP, regions_FN, rooms_TP,
                                     rooms_FP, rooms_FN]
            
            # gt visualize
            if 0:
                visualize_monte(unnormalized, (2, 999999, tgt), epoch, args.output_dir, batch_index + 1, target_d_rev, target_simple_cycles, target_results)

            all_points_TP += points_TP
            all_points_FP += points_FP
            all_points_FN += points_FN
            all_edges_TP += edges_TP
            all_edges_FP += edges_FP
            all_edges_FN += edges_FN
            all_regions_TP += regions_TP
            all_regions_FP += regions_FP
            all_regions_FN += regions_FN
            all_rooms_TP += rooms_TP
            all_rooms_FP += rooms_FP
            all_rooms_FN += rooms_FN
            if points_FP + points_FN + edges_FP + edges_FN == 0: # structure metric
                perfect_result_graph += 1
            if points_FP + points_FN + edges_FP + edges_FN + \
                regions_FP + regions_FN + rooms_FP + rooms_FN == 0: # overall metric
                perfect_result_plan += 1

        
            # print metrics
            count = batch_index + 1
            if count % 1 == 0:
                print('Epoch:', epoch, '\t',
                      str(count) + '/' + str(len(data_loader)), '\t',
                      'rooms_TP:', rooms_TP, '\t',
                      'rooms_FP:', rooms_FP, '\t',
                      'rooms_FN:', rooms_FN, 
                      )

    # calculate metrics in the evaluation
    if all_points_TP == 0:
        points_Prec = 0
    else:
        points_Prec = all_points_TP / (all_points_TP + all_points_FP)
    points_Rec = all_points_TP / (all_points_TP + all_points_FN)
    if all_edges_TP == 0:
        edges_Prec = 0
    else:
        edges_Prec = all_edges_TP / (all_edges_TP + all_edges_FP)
    edges_Rec = all_edges_TP / (all_edges_TP + all_edges_FN)
    if all_regions_TP == 0:
        regions_Prec = 0
    else:
        regions_Prec = all_regions_TP / (all_regions_TP + all_regions_FP)
    regions_Rec = all_regions_TP / (all_regions_TP + all_regions_FN)
    if all_rooms_TP == 0:
        rooms_Prec = 0
    else:
        rooms_Prec = all_rooms_TP / (all_rooms_TP + all_rooms_FP)
    rooms_Rec = all_rooms_TP / (all_rooms_TP + all_rooms_FN)


    end_time = datetime.now()

    # print metrics
    print('points_Prec:', format(points_Prec, '.4f'))
    print('points_Rec:', format(points_Rec, '.4f'))
    print('edges_Prec:', format(edges_Prec, '.4f'))
    print('edges_Rec:', format(edges_Rec, '.4f'))
    print('regions_Prec:', format(regions_Prec, '.4f'))
    print('regions_Rec:', format(regions_Rec, '.4f'))
    print('rooms_Prec:', format(rooms_Prec, '.4f'))
    print('rooms_Rec:', format(rooms_Rec, '.4f'))
    print('perfect_result_graph:', str(perfect_result_graph) + '/' + str(len(data_loader)))
    print('perfect_result_plan:', str(perfect_result_plan) + '/' + str(len(data_loader)))
    print('epoch eval_time:', time.strftime("%H:%M:%S", time.gmtime((end_time - start_time).seconds)))

    print('points F-1 score:', format(2 / ((1 / points_Prec) + (1 / points_Rec)), '.4f'))
    print('edges F-1 score:', format(2 / ((1 / edges_Prec) + (1 / edges_Rec)), '.4f'))
    print('regions F-1 score:', format(2 / ((1 / regions_Prec) + (1 / regions_Rec)), '.4f'))
    print('rooms F-1 score:', format(2 / ((1 / rooms_Prec) + (1 / rooms_Rec)), '.4f'))

    # log
    f = open(args.output_dir + '/val_iter_log.txt', mode='a')
    f_str = 'Epoch:' + str(epoch) + '\t' + \
            'points_Prec:' + str(format(points_Prec, '.4f')) + '\t' + \
            'points_Rec:' + str(format(points_Rec, '.4f')) + '\t' + \
            'edges_Prec:' + str(format(edges_Prec, '.4f')) + '\t' + \
            'edges_Rec:' + str(format(edges_Rec, '.4f')) + '\t' + \
            'regions_Prec:' + str(format(regions_Prec, '.4f')) + '\t' + \
            'regions_Rec:' + str(format(regions_Rec, '.4f')) + '\t' + \
            'rooms_Prec:' + str(format(rooms_Prec, '.4f')) + '\t' + \
            'rooms_Rec:' + str(format(rooms_Rec, '.4f')) + '\t' + \
            'perfect_result_graph:' + str(perfect_result_graph) + '/' + str(len(data_loader)) + '\t' + \
            'perfect_result_plan:' + str(perfect_result_plan) + '/' + str(len(data_loader)) + '\t' + \
            'epoch eval_time:' + time.strftime("%H:%M:%S", time.gmtime((end_time - start_time).seconds)) + '\n'
    f.write(f_str)
    f.close()
