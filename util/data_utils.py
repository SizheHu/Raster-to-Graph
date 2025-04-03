import copy
import random
import time

import cv2
import torch
import numpy as np
from PIL import Image

from util.edges_utils import get_edges_alldirections, get_edges_alldirections_rev
from util.math_utils import clip
from util.mean_std import mean, std


def data_to_cuda(samples, targets):
    return samples.to(torch.device('cuda')), [{k: v.to(torch.device('cuda')) for k, v in t.items()} for t in targets]


def get_random_layer_targets(targets, gt_layer):
    random_targets = []
    for batch_i, target_i in enumerate(targets):
        random_layer_targets_i = copy.deepcopy(target_i)
        if gt_layer[batch_i] != len(random_layer_targets_i['layer_indices']) - 1:
            start = random_layer_targets_i['layer_indices'][gt_layer[batch_i]].item()
            end = random_layer_targets_i['layer_indices'][gt_layer[batch_i] + 1].item()
        else:
            start = random_layer_targets_i['layer_indices'][gt_layer[batch_i]].item()
            end = len(random_layer_targets_i['points'])
        random_points_i = random_layer_targets_i['points'][start:end, :]
        random_edges_i = random_layer_targets_i['edges'][start:end]
        random_unnormalized_points_i = random_layer_targets_i['unnormalized_points'][start:end, :]
        random_layer_targets_i['points'] = random_points_i
        random_layer_targets_i['edges'] = random_edges_i
        random_layer_targets_i['unnormalized_points'] = random_unnormalized_points_i
        del random_layer_targets_i['layer_indices']
        random_targets.append(random_layer_targets_i)
    return random_targets


def random_layers(targets):
    return [random.randint(0, len(targets[i]['layer_indices']) - 1) for i in range(len(targets))]

def get_given_layers_random_region(targets, graphs):
    random_regions = []
    for bs_i in range(len(targets)):
        # target
        targets_i = targets[bs_i]
        graphs_i = graphs[bs_i]
        # level 0: start
        start_i = tuple(targets_i['unnormalized_points'][0].tolist())

        # sampled prob: for a neighborhood, each node is sampled by this probability
        # sampled_prob = 0.0001
        # sampled_prob = random.random()
        sampled_prob = 0.5
        # sampled_prob = 1

        # sampled nodes
        sampled_points = {}
        for point_tensor in targets_i['unnormalized_points']:
            pos = tuple(point_tensor.tolist())
            sampled_points[pos] = 0
        # edges of sampled nodes
        sampled_edges = []

        # nodes number of subgraph
        sampled_amount = random.randint(0, len(sampled_points) + 2)
        if sampled_amount in [len(sampled_points) + 1]:
            sampled_amount = 0
        if sampled_amount in [len(sampled_points) + 2]:
            sampled_amount = len(sampled_points)
        # sampled_amount = random.randint(0, len(sampled_points))


        # Note that when sampled_prob = 1, the number of sampled nodes must be in 'layer_indices' or be the total number of points to ensure that the entire layers is sampled.
        # equal to BFS
        if sampled_prob == 1:
            l = targets_i['layer_indices'].tolist()
            l.append(len(sampled_points))
            l.append(0)
            l.append(len(sampled_points))
            sampled_amount = l[random.randint(0, len(l) - 1)]


        # start sampling
        if sampled_amount == 0:
            random_regions.append((sampled_points, sampled_edges))
            continue
        sampled_points[start_i] = 1
        if sampled_amount == 1:
            random_regions.append((sampled_points, sampled_edges))
            continue
        while sum(sampled_points.values()) < sampled_amount:
            all_sampled_points = set([k for k, v in sampled_points.items() if v == 1])
            all_sampled_points_adjs = set()
            for sampled_point in all_sampled_points:
                adj = set(graphs_i[sampled_point])
                all_sampled_points_adjs = all_sampled_points_adjs.union(adj)
            all_sampled_points_adjs.remove((-1, -1))
            all_sampled_points_adjs = list(all_sampled_points_adjs.difference(all_sampled_points))
            # shuffle the last layer to let it uniform (no bias of sample order)
            random.shuffle(all_sampled_points_adjs)
            # determine whether to sample nodes in each neighborhood based on probability
            for all_sampled_points_adj_index, all_sampled_points_adj in enumerate(all_sampled_points_adjs):
                all_sampled_points = set([k for k, v in sampled_points.items() if v == 1])
                if sum(sampled_points.values()) == sampled_amount:
                    break
                else:
                    if 1:
                        if random.random() < sampled_prob:
                            sampled_points[all_sampled_points_adj] = 1
                            # sample edges
                            all_pos1s = graphs_i[all_sampled_points_adj]
                            pos2 = all_sampled_points_adj
                            for pos1 in all_pos1s:
                                if pos1 in all_sampled_points:
                                    sampled_edges.append((pos1, pos2))
                        else:
                            sampled_points[all_sampled_points_adj] = 0
        random_regions.append((sampled_points, sampled_edges))
    return random_regions


def get_random_region_targets(given_layers, graphs, targets):
    random_region_targets = []
    for bs_i in range(len(targets)):
        random_region_target = {}
        targets_i = targets[bs_i]
        graphs_i = graphs[bs_i]
        given_layers_i = given_layers[bs_i]
        sampled_points_i, sampled_edges_i = given_layers_i

        if sum(sampled_points_i.values()) == 0:
            random_region_target['edges'] = targets_i['edges'][:1]

            random_region_target['semantic_left_up'] = targets_i['semantic_left_up'][:1]
            random_region_target['semantic_right_up'] = targets_i['semantic_right_up'][:1]
            random_region_target['semantic_right_down'] = targets_i['semantic_right_down'][:1]
            random_region_target['semantic_left_down'] = targets_i['semantic_left_down'][:1]

            random_region_target['image_id'] = targets_i['image_id']
            random_region_target['size'] = targets_i['size']
            random_region_target['unnormalized_points'] = targets_i['unnormalized_points'][:1]
            random_region_target['points'] = targets_i['points'][:1]
            random_region_target['last_edges'] = torch.zeros((1,), dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_target['this_edges'] = torch.zeros((1,), dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_targets.append(random_region_target)
        elif 1 <= sum(sampled_points_i.values()) <= len(sampled_points_i) - 1:
            sampled_points_i_given = set([k for k, v in sampled_points_i.items() if v == 1])
            unnormalized_points = []
            for point, sampled_or_not in sampled_points_i.items():
                if sampled_or_not == 0:
                    adjs = graphs_i[point]
                    for adj in adjs:
                        if adj in sampled_points_i_given:
                            unnormalized_points.append(point)
                            break

            indices_for_semantic = []
            for unnormalized_point in unnormalized_points:
                for ind, every_point in enumerate(targets_i['unnormalized_points']):
                    every_point = tuple(every_point.tolist())
                    if abs(every_point[0] - unnormalized_point[0]) <= 2 and abs(every_point[1] - unnormalized_point[1]) <= 2:
                        indices_for_semantic.append(ind)
            assert len(unnormalized_points) == len(indices_for_semantic)
            semantic_left_up = []
            semantic_right_up = []
            semantic_right_down = []
            semantic_left_down = []
            for ind in indices_for_semantic:
                semantic_left_up.append(targets_i['semantic_left_up'][ind].item())
                semantic_right_up.append(targets_i['semantic_right_up'][ind].item())
                semantic_right_down.append(targets_i['semantic_right_down'][ind].item())
                semantic_left_down.append(targets_i['semantic_left_down'][ind].item())

            edges = []
            for unnormalized_point in unnormalized_points:
                edge = ''
                adjs = graphs_i[unnormalized_point]
                for adj in adjs:
                    if adj != (-1, -1):
                        edge += '1'
                    else:
                        edge += '0'
                edge = get_edges_alldirections_rev(edge)
                edges.append(edge)
            last_edges = []
            for unnormalized_point in unnormalized_points:
                last_edge = ''
                adjs = graphs_i[unnormalized_point]
                for adj in adjs:
                    if adj in sampled_points_i_given:
                        last_edge += '1'
                    else:
                        last_edge += '0'
                last_edge = get_edges_alldirections_rev(last_edge)
                last_edges.append(last_edge)
            this_edges = []
            for unnormalized_point in unnormalized_points:
                this_edge = ''
                adjs = graphs_i[unnormalized_point]
                for adj in adjs:
                    if adj in unnormalized_points:
                        this_edge += '1'
                    else:
                        this_edge += '0'
                this_edge = get_edges_alldirections_rev(this_edge)
                this_edges.append(this_edge)

            random_region_target['edges'] = torch.tensor(edges, dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)


            random_region_target['semantic_left_up'] = torch.tensor(semantic_left_up, dtype=targets_i['semantic_left_up'].dtype,
                                                                    device=targets_i['semantic_left_up'].device)
            random_region_target['semantic_right_up'] = torch.tensor(semantic_right_up, dtype=targets_i['semantic_right_up'].dtype,
                                                                    device=targets_i['semantic_right_up'].device)
            random_region_target['semantic_right_down'] = torch.tensor(semantic_right_down, dtype=targets_i['semantic_right_down'].dtype,
                                                                    device=targets_i['semantic_right_down'].device)
            random_region_target['semantic_left_down'] = torch.tensor(semantic_left_down, dtype=targets_i['semantic_left_down'].dtype,
                                                                    device=targets_i['semantic_left_down'].device)


            random_region_target['image_id'] = targets_i['image_id']
            random_region_target['size'] = targets_i['size']
            random_region_target['unnormalized_points'] = torch.tensor(unnormalized_points,
                                                                       dtype=targets_i['unnormalized_points'].dtype,
                                                                       device=targets_i['unnormalized_points'].device)
            random_region_target['points'] = torch.tensor(unnormalized_points,
                                                          dtype=targets_i['points'].dtype,
                                                          device=targets_i['points'].device) / targets_i['size']
            random_region_target['last_edges'] = torch.tensor(last_edges, dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_target['this_edges'] = torch.tensor(this_edges, dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_targets.append(random_region_target)
        else:
            random_region_target['edges'] = 16 * torch.ones(targets_i['edges'][:1].shape, dtype=targets_i['edges'].dtype,
                                                            device=targets_i['edges'].device)

            random_region_target['semantic_left_up'] = 11 * torch.ones(targets_i['semantic_left_up'][:1].shape,
                                                                       dtype=targets_i['semantic_left_up'].dtype,
                                                                        device=targets_i['semantic_left_up'].device)
            random_region_target['semantic_right_up'] = 11 * torch.ones(targets_i['semantic_right_up'][:1].shape,
                                                                       dtype=targets_i['semantic_right_up'].dtype,
                                                                       device=targets_i['semantic_right_up'].device)
            random_region_target['semantic_right_down'] = 11 * torch.ones(targets_i['semantic_right_down'][:1].shape,
                                                                       dtype=targets_i['semantic_right_down'].dtype,
                                                                       device=targets_i['semantic_right_down'].device)
            random_region_target['semantic_left_down'] = 11 * torch.ones(targets_i['semantic_left_down'][:1].shape,
                                                                       dtype=targets_i['semantic_left_down'].dtype,
                                                                       device=targets_i['semantic_left_down'].device)

            random_region_target['image_id'] = targets_i['image_id']
            random_region_target['size'] = targets_i['size']
            random_region_target['unnormalized_points'] = 505 * torch.ones(targets_i['unnormalized_points'][:1].shape,
                                                                           dtype=targets_i['unnormalized_points'][:1].dtype,
                                                                           device=targets_i['unnormalized_points'][:1].device)
            random_region_target['points'] = (505 * torch.ones(targets_i['unnormalized_points'][:1].shape,
                                                               dtype=targets_i['points'][:1].dtype,
                                                               device=targets_i['points'][:1].device)) / targets_i['size']
            random_region_target['last_edges'] = 16 * torch.ones((1,), dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_target['this_edges'] = 16 * torch.ones((1,), dtype=targets_i['edges'].dtype, device=targets_i['edges'].device)
            random_region_targets.append(random_region_target)

    return random_region_targets


def random_pertubation(sampled_points_i, sampled_edges_i):
    random_pertube_map = {}
    sigma = 2
    pertube_threshold = 5
    for sampled_point in sampled_points_i:
        random_pertube_map[sampled_point] = (sampled_point[0] + clip(int(random.gauss(0, sigma)), -1 * pertube_threshold, pertube_threshold),
                                           sampled_point[1] + clip(int(random.gauss(0, sigma)), -1 * pertube_threshold, pertube_threshold))
    new_sampled_points_i = {}
    new_sampled_edges_i = []
    for sampled_point in sampled_points_i:
        new_sampled_points_i[random_pertube_map[sampled_point]] = sampled_points_i[sampled_point]
    for (pos1, pos2) in sampled_edges_i:
        new_sampled_edges_i.append((random_pertube_map[pos1], random_pertube_map[pos2]))
    return new_sampled_points_i, new_sampled_edges_i


def draw_given_layers_on_tensors_random_region(given_layers, tensors, graphs):
    '''draw 9*9 yellow squares and width 2 blue lines'''
    tensors_list = []
    unnormalized_list = []
    for i in range(len(given_layers)):
        temp_tensor = tensors[i]

        temp_tensor_0 = (temp_tensor[0] * std[0] + mean[0]) * 255
        temp_tensor_1 = (temp_tensor[1] * std[1] + mean[1]) * 255
        temp_tensor_2 = (temp_tensor[2] * std[2] + mean[2]) * 255

        rectangle_radius = 5

        # end sign
        endsign = (505, 505)
        valid_violet_endsign_up = endsign[1] - rectangle_radius
        valid_violet_endsign_down = endsign[1] + rectangle_radius
        valid_violet_endsign_left = endsign[0] - rectangle_radius
        valid_violet_endsign_right = endsign[0] + rectangle_radius
        temp_tensor_0[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 255
        temp_tensor_1[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 0
        temp_tensor_2[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 255

        sampled_points_i, sampled_edges_i = given_layers[i]
        sampled_points_i, sampled_edges_i = random_pertubation(sampled_points_i, sampled_edges_i)

        given_points = [k for k, v in sampled_points_i.items() if v == 1]

        for j, pos in enumerate(given_points):
            valid_yellow_pos_up = (pos[1] - rectangle_radius) if (pos[1] - rectangle_radius) >= 0 else 0
            valid_yellow_pos_down = (pos[1] + rectangle_radius) if (pos[1] + rectangle_radius) < temp_tensor.shape[2] else temp_tensor.shape[
                                                                                                                               2] - 1
            valid_yellow_pos_left = (pos[0] - rectangle_radius) if (pos[0] - rectangle_radius) >= 0 else 0
            valid_yellow_pos_right = (pos[0] + rectangle_radius) if (pos[0] + rectangle_radius) < temp_tensor.shape[1] else temp_tensor.shape[
                                                                                                                                1] - 1
            temp_tensor_0[valid_yellow_pos_up:valid_yellow_pos_down + 1, valid_yellow_pos_left:valid_yellow_pos_right + 1] = 255
            temp_tensor_1[valid_yellow_pos_up:valid_yellow_pos_down + 1, valid_yellow_pos_left:valid_yellow_pos_right + 1] = 255
            temp_tensor_2[valid_yellow_pos_up:valid_yellow_pos_down + 1, valid_yellow_pos_left:valid_yellow_pos_right + 1] = 0

        # draw blue lines
        line_width = 2
        for edge in sampled_edges_i:
            pos1 = edge[0]
            pos2 = edge[1]
            if abs(pos1[0] - pos2[0]) < abs(pos1[1] - pos2[1]):
                if pos1[1] > pos2[1]:
                    temp_tensor_0[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 255
                else:
                    temp_tensor_0[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 255
            else:
                if pos1[0] > pos2[0]:
                    temp_tensor_0[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_1[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_2[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 255
                else:
                    temp_tensor_0[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_1[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_2[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 255

        unnormalized = torch.stack((temp_tensor_0, temp_tensor_1, temp_tensor_2), dim=0)
        unnormalized_list.append(unnormalized)

        temp_tensor_0_renorm = ((temp_tensor_0 / 255) - mean[0]) / std[0]
        temp_tensor_1_renorm = ((temp_tensor_1 / 255) - mean[1]) / std[1]
        temp_tensor_2_renorm = ((temp_tensor_2 / 255) - mean[2]) / std[2]

        temp_tensor = torch.stack([temp_tensor_0_renorm, temp_tensor_1_renorm, temp_tensor_2_renorm], dim=0)

        tensors_list.append(temp_tensor)

    return torch.stack(tensors_list, dim=0), torch.stack(unnormalized_list, dim=0)


def initialize_tensors(tensors):
    tensors_list = []
    unnormalized_list = []
    for i in range(len(tensors)):
        temp_tensor = tensors[i]

        temp_tensor_0 = (temp_tensor[0] * std[0] + mean[0]) * 255
        temp_tensor_1 = (temp_tensor[1] * std[1] + mean[1]) * 255
        temp_tensor_2 = (temp_tensor[2] * std[2] + mean[2]) * 255

        rectangle_radius = 5 # 4+1+4=9

        # end sign (when predict this, AR iteration terminates)
        endsign = (505, 505)
        valid_violet_endsign_up = endsign[1] - rectangle_radius
        valid_violet_endsign_down = endsign[1] + rectangle_radius
        valid_violet_endsign_left = endsign[0] - rectangle_radius
        valid_violet_endsign_right = endsign[0] + rectangle_radius
        temp_tensor_0[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 255
        temp_tensor_1[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 0
        temp_tensor_2[valid_violet_endsign_up:valid_violet_endsign_down + 1, valid_violet_endsign_left:valid_violet_endsign_right + 1] = 255

        unnormalized = torch.stack((temp_tensor_0, temp_tensor_1, temp_tensor_2), dim=0)
        unnormalized_list.append(unnormalized)

        temp_tensor_0_renorm = ((temp_tensor_0 / 255) - mean[0]) / std[0]
        temp_tensor_1_renorm = ((temp_tensor_1 / 255) - mean[1]) / std[1]
        temp_tensor_2_renorm = ((temp_tensor_2 / 255) - mean[2]) / std[2]

        temp_tensor = torch.stack([temp_tensor_0_renorm, temp_tensor_1_renorm, temp_tensor_2_renorm], dim=0)

        tensors_list.append(temp_tensor)

    return torch.stack(tensors_list, dim=0), torch.stack(unnormalized_list, dim=0)


def l1_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def delete_graphs(targets):
    no_graph_targets = []
    for target in targets:
        target_ = copy.deepcopy(target)
        del target_['graph']
        no_graph_targets.append(target_)
    return no_graph_targets


def delete_graphs_and_unnormpoints(targets):
    no_graph_targets = []
    for target in targets:
        target_ = copy.deepcopy(target)
        del target_['graph']
        del target_['unnormalized_points']
        no_graph_targets.append(target_)
    return no_graph_targets


def get_remove_point(this_preds, dist_threshold):
    for point1 in this_preds:
        for point2 in this_preds:
            # if point1 != point2:
            if not ((point1['points'].tolist()[0] == point2['points'].tolist()[0]) and (point1['points'].tolist()[1] == point2['points'].tolist()[1])):
                dist_chebyshev = max(abs(point1['points'].tolist()[0] - point2['points'].tolist()[0]),
                                     abs(point1['points'].tolist()[1] - point2['points'].tolist()[1]))
                if dist_chebyshev <= dist_threshold:
                    point1_confidence = point1['scores'].item()
                    point2_confidence = point2['scores'].item()
                    if point1_confidence < point2_confidence:
                        return point1
                    elif point2_confidence < point1_confidence:
                        return point2
                    else:
                        return [point1, point2][random.randint(0, 1)]
    return None

def point_inside(point, points_list):
    point1 = tuple(point['points'].tolist())
    for point_i in points_list:
        point1_i = tuple(point_i['points'].tolist())
        if point1 == point1_i:
            return True
    return False

def remove_points(need_to_remove_in_last_edges, this_preds):
    result = []
    for this_pred in this_preds:
        if not point_inside(this_pred, need_to_remove_in_last_edges):
            result.append(this_pred)
    return result

def nms(this_preds):
    if len(this_preds) <= 1:
        return this_preds
    else:
        dist_threshold = 5
        while True:
            remove_point = get_remove_point(this_preds, dist_threshold)
            if remove_point is None:
                break
            else:
                # this_preds.remove(remove_point)
                this_preds = remove_points([remove_point], this_preds)

        return this_preds

def nms_givenpoints(this_preds, preds):
    if len(this_preds) == 0:
        return this_preds
    else:
        all_given_points = []
        for (given_points, given_last_edges, given_this_edges) in preds:
            all_given_points.extend(given_points)
        if len(all_given_points) == 0:
            return this_preds
        this_preds_copy = copy.deepcopy(this_preds)
        dist_threshold = 5
        for this_pred in this_preds_copy:
            for given_point in all_given_points:
                this_pred_pos = tuple(this_pred['points'].tolist())
                given_point_pos = tuple(given_point['points'].tolist())
                dist_chebyshev = max(abs(this_pred_pos[0] - given_point_pos[0]),
                                     abs(this_pred_pos[1] - given_point_pos[1]))
                if dist_chebyshev <= dist_threshold:
                    this_preds = remove_points([this_pred], this_preds)
                    break
        return this_preds


def random_keep(this_preds):
    if len(this_preds) <= 1:
        return this_preds
    else:
        while True:
            random_keep_this_preds = []
            for point in this_preds:
                # is_keep = random.random() < point['scores'].item()
                is_keep = random.random() < 1.01
                # is_keep = random.random() < 0.5
                if is_keep:
                    random_keep_this_preds.append(point)
            if len(random_keep_this_preds) > 0:
                return random_keep_this_preds


def is_stop(this_preds):
    if len(this_preds) == 0:
        return 1 # stop
    elif (len(this_preds) >= 1) and (16 in [p['edges'].item() for p in this_preds]):
        return 2 # normally terminate
    else:
        return 0 # not stop


def draw_preds_on_tensors(preds, tensors):
    tensors_list = []
    unnormalized_list = []

    for i in range(len(tensors)):
        temp_tensor = tensors[i]

        temp_tensor_0 = (temp_tensor[0] * std[0] + mean[0]) * 255
        temp_tensor_1 = (temp_tensor[1] * std[1] + mean[1]) * 255
        temp_tensor_2 = (temp_tensor[2] * std[2] + mean[2]) * 255

        rectangle_radius = 5

        (this_preds, last_edges, this_edges) = preds[-1]
        for this_pred in this_preds:
            point = tuple([int(_) for _ in this_pred['points'].tolist()])
            up = point[1] - rectangle_radius
            down = point[1] + rectangle_radius
            left = point[0] - rectangle_radius
            right = point[0] + rectangle_radius
            temp_tensor_0[up:down + 1, left:right + 1] = 255
            temp_tensor_1[up:down + 1, left:right + 1] = 255
            temp_tensor_2[up:down + 1, left:right + 1] = 0
        line_width = 2
        for last_edge in last_edges:
            pos1 = tuple([int(_) for _ in last_edge[0]['points'].tolist()])
            pos2 = tuple([int(_) for _ in last_edge[1]['points'].tolist()])
            if abs(pos1[0] - pos2[0]) < abs(pos1[1] - pos2[1]):
                if pos1[1] > pos2[1]:
                    temp_tensor_0[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 255
                else:
                    temp_tensor_0[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 255
            else:
                if pos1[0] > pos2[0]:
                    temp_tensor_0[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_1[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_2[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 255
                else:
                    temp_tensor_0[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_1[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_2[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 255
        for this_edge in this_edges:
            pos1 = tuple([int(_) for _ in this_edge[0]['points'].tolist()])
            pos2 = tuple([int(_) for _ in this_edge[1]['points'].tolist()])
            if abs(pos1[0] - pos2[0]) < abs(pos1[1] - pos2[1]):
                if pos1[1] > pos2[1]:
                    temp_tensor_0[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos2[1]: pos1[1] + 1, int((pos1[0] + pos2[0]) / 2) - int(line_width / 2): int((pos1[0] + pos2[0]) / 2) + int(line_width / 2) + 1] = 255
                else:
                    temp_tensor_0[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_1[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 0
                    temp_tensor_2[pos1[1]: pos2[1] + 1, int((pos2[0] + pos1[0]) / 2) - int(line_width / 2): int((pos2[0] + pos1[0]) / 2) + int(line_width / 2) + 1] = 255
            else:
                if pos1[0] > pos2[0]:
                    temp_tensor_0[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_1[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 0
                    temp_tensor_2[int((pos1[1] + pos2[1]) / 2) - int(line_width / 2): int((pos1[1] + pos2[1]) / 2) + int(line_width / 2) + 1, pos2[0]: pos1[0] + 1] = 255
                else:
                    temp_tensor_0[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_1[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 0
                    temp_tensor_2[int((pos2[1] + pos1[1]) / 2) - int(line_width / 2): int((pos2[1] + pos1[1]) / 2) + int(line_width / 2) + 1, pos1[0]: pos2[0] + 1] = 255


        unnormalized = torch.stack((temp_tensor_0, temp_tensor_1, temp_tensor_2), dim=0)
        unnormalized_list.append(unnormalized)

        temp_tensor_0_renorm = ((temp_tensor_0 / 255) - mean[0]) / std[0]
        temp_tensor_1_renorm = ((temp_tensor_1 / 255) - mean[1]) / std[1]
        temp_tensor_2_renorm = ((temp_tensor_2 / 255) - mean[2]) / std[2]

        temp_tensor = torch.stack([temp_tensor_0_renorm, temp_tensor_1_renorm, temp_tensor_2_renorm], dim=0)

        tensors_list.append(temp_tensor)

    return torch.stack(tensors_list, dim=0), torch.stack(unnormalized_list, dim=0)

def edge_inside(edge, edges_list):
    edge_point1 = tuple(edge[0]['points'].tolist())
    edge_point2 = tuple(edge[1]['points'].tolist())
    for edge_i in edges_list:
        edge_i_point1 = tuple(edge_i[0]['points'].tolist())
        edge_i_point2 = tuple(edge_i[1]['points'].tolist())
        if ((edge_point1 == edge_i_point1) and (edge_point2 == edge_i_point2)) or \
            ((edge_point1 == edge_i_point2) and (edge_point2 == edge_i_point1)):
            return True
    return False

def remove_edge(edge, edges_list):
    result = []
    edge_point1 = tuple(edge[0]['points'].tolist())
    edge_point2 = tuple(edge[1]['points'].tolist())
    for edge_i in edges_list:
        edge_i_point1 = tuple(edge_i[0]['points'].tolist())
        edge_i_point2 = tuple(edge_i[1]['points'].tolist())
        if (edge_point1 == edge_i_point1) and (edge_point2 == edge_i_point2):
            pass
        else:
            result.append(edge_i)
    return result

def get_edges_amount(preds):
    count = 0
    for (this_preds, last_edges, this_edges) in preds:
        count += len(last_edges)
        count += len(this_edges)
    return count

def get_reserve_preds(results, keep_confidence_threshold, targets):
    reserve_preds = []

    valid_label_indices_edges = torch.where(results['edges'] != 0)[0]
    valid_label_indices_scores = torch.where(results['scores'] <= keep_confidence_threshold)[0]
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
        reserve_preds.append(valid_results_i)
    return reserve_preds
