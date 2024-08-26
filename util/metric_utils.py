import copy
import math
from util.geom_utils import *


def calculate_AP(valid_results, ground_truths, confidence_final):
    ground_truths_copy = copy.deepcopy(ground_truths)
    all_preds = []
    for image_id, image_pred in valid_results.items():
        for i in range(len(image_pred['points'])):
            pred = {}
            pred['score'] = image_pred['scores'][i].item()
            pred['point'] = tuple(image_pred['points'][i].tolist())
            pred['size'] = tuple(image_pred['size'].tolist())
            pred['image_id'] = image_id.item()
            all_preds.append(pred)
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)

    all_preds = [pred for pred in all_preds if pred['score'] > confidence_final]

    all_metrics = []
    for n in range(1, len(all_preds) + 1):

        ground_truths = copy.deepcopy(ground_truths_copy)

        sub_preds = all_preds[0:n]

        TP = 0
        FP = 0
        FN = 0
        for pred in sub_preds:
            pred_point = pred['point']
            img_size = (pred['size'][1], pred['size'][0])
            img_id = pred['image_id']
            dist_threshold = (img_size[0] * 0.01, img_size[1] * 0.01)
            gt = [tuple(gt_point) for gt_point in ground_truths[img_id]['points'].tolist()]
            gt_copy = copy.deepcopy(gt)
            euc_dists = {}
            dists = {}
            for gt_point in gt_copy:
                if gt_point[2] == 0:
                    dist = (abs(pred_point[0] - gt_point[0]), abs(pred_point[1] - gt_point[1]))
                    euc_dist = math.sqrt(dist[0] ** 2 + dist[1] ** 2)
                    euc_dists[gt_point] = euc_dist
                    dists[gt_point] = dist
            euc_dists = sorted(euc_dists.items(), key=lambda x: x[1])
            if len(euc_dists) == 0:
                FP += 1
                continue
            nearest_gt_point = euc_dists[0][0]
            min_dist = dists[nearest_gt_point]
            if min_dist[0] < dist_threshold[0] and min_dist[1] < dist_threshold[1]:
                gtip = ground_truths[img_id]['points']
                for i, p in enumerate(gtip):
                    if p[0].item() == nearest_gt_point[0] and \
                            p[1].item() == nearest_gt_point[1] and \
                            p[2].item() == nearest_gt_point[2]:
                        # print('qqq', p, nearest_gt_point)
                        gtip[i, 2] = 1
                        break
                ground_truths[img_id]['points'] = gtip
                # print('rrr', ground_truths[img_id]['points'])
                TP += 1
                continue
            FP += 1
        for img_id, points in ground_truths.items():
            points = points['points']
            for point in points:
                if point[2] == 0:
                    FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        # print(n, TP, FP, FN, precision, recall)
        all_metrics.append((precision, recall))

    all_metrics = sorted(all_metrics, key=lambda x: (x[1], x[0]))
    p_r_curve_points = {}
    for point in all_metrics:
        p_r_curve_points[point[1]] = point[0]
    p_r_curve_points[0] = 1
    p_r_curve_points = sorted(p_r_curve_points.items(), key=lambda d: d[0])
    AP = 0
    for i, rp in enumerate(p_r_curve_points):
        r = rp[0]
        p = rp[1]
        if i > 0:
            small_rectangular_area = (r - p_r_curve_points[i - 1][0]) * p
            AP += small_rectangular_area

    return AP


def get_results(best_result):
    if 1:
        preds = best_result[2]
        output_points = []
        output_edges = []
        for triplet in preds:
            this_preds = triplet[0]
            last_edges = triplet[1]
            this_edges = triplet[2]
            for this_pred in this_preds:
                point = tuple(this_pred['points'].int().tolist())
                output_points.append(point)
            for last_edge in last_edges:
                point1 = tuple(last_edge[0]['points'].int().tolist())
                point2 = tuple(last_edge[1]['points'].int().tolist())
                edge = (point1, point2)
                output_edges.append(edge)
            for this_edge in this_edges:
                point1 = tuple(this_edge[0]['points'].int().tolist())
                point2 = tuple(this_edge[1]['points'].int().tolist())
                edge = (point1, point2)
                output_edges.append(edge)
        return output_points, output_edges


def get_results_visual(best_result):
    if 1:
        preds = best_result[2]
        output_points = []
        output_edges = []
        for layer_index, triplet in enumerate(preds):
            this_preds = triplet[0]
            last_edges = triplet[1]
            this_edges = triplet[2]
            for this_pred in this_preds:
                point = tuple(this_pred['points'].int().tolist())
                output_points.append([layer_index, point])
            for last_edge in last_edges:
                point1 = tuple(last_edge[0]['points'].int().tolist())
                point2 = tuple(last_edge[1]['points'].int().tolist())
                edge = (point1, point2)
                output_edges.append([layer_index, edge])
            for this_edge in this_edges:
                point1 = tuple(this_edge[0]['points'].int().tolist())
                point2 = tuple(this_edge[1]['points'].int().tolist())
                edge = (point1, point2)
                output_edges.append([layer_index, edge])
        return output_points, output_edges, len(preds)

def get_results_float_with_semantic(best_result):
    if 1:
        preds = best_result[2]
        output_points = []
        output_edges = []
        for triplet in preds:
            this_preds = triplet[0]
            last_edges = triplet[1]
            this_edges = triplet[2]
            for this_pred in this_preds:
                point = (this_pred['points'].tolist()[0], this_pred['points'].tolist()[1],
                         this_pred['semantic_left_up'].item(), this_pred['semantic_right_up'].item(),
                         this_pred['semantic_right_down'].item(), this_pred['semantic_left_down'].item())
                output_points.append(point)
            for last_edge in last_edges:
                point1 = (last_edge[0]['points'].tolist()[0], last_edge[0]['points'].tolist()[1],
                         last_edge[0]['semantic_left_up'].item(), last_edge[0]['semantic_right_up'].item(),
                         last_edge[0]['semantic_right_down'].item(), last_edge[0]['semantic_left_down'].item())
                point2 = (last_edge[1]['points'].tolist()[0], last_edge[1]['points'].tolist()[1],
                          last_edge[1]['semantic_left_up'].item(), last_edge[1]['semantic_right_up'].item(),
                          last_edge[1]['semantic_right_down'].item(), last_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
            for this_edge in this_edges:
                point1 = (this_edge[0]['points'].tolist()[0], this_edge[0]['points'].tolist()[1],
                          this_edge[0]['semantic_left_up'].item(), this_edge[0]['semantic_right_up'].item(),
                          this_edge[0]['semantic_right_down'].item(), this_edge[0]['semantic_left_down'].item())
                point2 = (this_edge[1]['points'].tolist()[0], this_edge[1]['points'].tolist()[1],
                          this_edge[1]['semantic_left_up'].item(), this_edge[1]['semantic_right_up'].item(),
                          this_edge[1]['semantic_right_down'].item(), this_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
        return output_points, output_edges



def calculate_single_sample(best_result, graph, target_d_rev, target_simple_cycles, target_results, d_rev, simple_cycles, results):
    output_points, output_edges = get_results(best_result)
    gt_points = [k for k, v in graph.items()]
    gt_edges = []
    for k, v in graph.items():
        for adj in v:
            if adj != (-1, -1):
                gt_edge = (k, adj)
                if (adj, k) not in gt_edges:
                    gt_edges.append(gt_edge)

    points_TP = 0
    points_FP = 0
    points_FN = 0
    dist_error_x = 0
    dist_error_y = 0
    dist_error_l2 = 0
    gt_points_copy = copy.deepcopy(gt_points)
    threshold = 5
    for output_point in output_points:
        matched = False
        for gt_point in gt_points:
            if (abs(output_point[0] - gt_point[0]) <= threshold) and (abs(output_point[1] - gt_point[1]) <= threshold):
                if gt_point in gt_points_copy:
                    points_TP += 1
                    dist_error_x += abs(output_point[0] - gt_point[0])
                    dist_error_y += abs(output_point[1] - gt_point[1])
                    dist_error_l2 += (abs(output_point[0] - gt_point[0]) ** 2 + abs(output_point[1] - gt_point[1]) ** 2) ** 0.5
                    matched = True
                    gt_points_copy.remove(gt_point)
                    break
        if not matched:
            points_FP += 1
    points_FN = len(gt_points) - points_TP

    edges_TP = 0
    edges_FP = 0
    edges_FN = 0
    gt_edges_copy = copy.deepcopy(gt_edges)
    threshold = 5
    for output_edge in output_edges:
        matched = False
        for gt_edge in gt_edges:
            if (((abs(output_edge[0][0] - gt_edge[0][0]) <= threshold) and (abs(output_edge[0][1] - gt_edge[0][1]) <= threshold)) and
                ((abs(output_edge[1][0] - gt_edge[1][0]) <= threshold) and (abs(output_edge[1][1] - gt_edge[1][1]) <= threshold))) or \
                    (((abs(output_edge[0][0] - gt_edge[1][0]) <= threshold) and (abs(output_edge[0][1] - gt_edge[1][1]) <= threshold)) and
                     ((abs(output_edge[1][0] - gt_edge[0][0]) <= threshold) and (abs(output_edge[1][1] - gt_edge[0][1]) <= threshold))):
                if gt_edge in gt_edges_copy:
                    edges_TP += 1
                    matched = True
                    gt_edges_copy.remove(gt_edge)
                    break
        if not matched:
            edges_FP += 1
    edges_FN = len(gt_edges) - edges_TP


    regions_TP = 0
    regions_FP = 0
    regions_FN = 0
    rooms_TP = 0
    rooms_FP = 0
    rooms_FN = 0
    gt_regions = []
    output_regions = []

    for target_simple_cycle in target_simple_cycles:
        target_polyg = [(point_i[0], point_i[1])
                 for point_i in target_simple_cycle]
        gt_regions.append(target_polyg)

    for simple_cycle in simple_cycles:
        polyg = [(point_i[0], point_i[1])
                 for point_i in simple_cycle]
        polyg.pop(-1)
        output_regions.append(polyg)
    gt_regions_copy = copy.deepcopy(gt_regions)
    iou_threshold = 0.7
    for output_region_i, output_region in enumerate(output_regions):
        matched = False
        for gt_region_i, gt_region in enumerate(gt_regions):
            # print(output_region)
            # print(gt_region)
            # print(poly_iou(Polygon(gt_region), Polygon(output_region)))
            if poly_iou(Polygon(gt_region), Polygon(output_region)) >= iou_threshold:
                if gt_region in gt_regions_copy:
                    regions_TP += 1
                    if target_results[gt_region_i] == results[output_region_i]:
                        rooms_TP += 1
                    else:
                        rooms_FP += 1
                    matched = True
                    gt_regions_copy.remove(gt_region)
                    break
        if not matched:
            regions_FP += 1
            rooms_FP += 1
    regions_FN = len(gt_regions) - regions_TP
    rooms_FN = len(gt_regions) - rooms_TP
    # print(regions_TP, regions_FP, regions_FN)
    # print(rooms_TP, rooms_FP, rooms_FN)

    dist_error = (0, 0, 0)
    if points_TP > 0:
        dist_error = (dist_error_x, dist_error_y, dist_error_l2)
    return points_TP, points_FP, points_FN, edges_TP, edges_FP, edges_FN, \
        dist_error, regions_TP, regions_FP, regions_FN, rooms_TP, rooms_FP, rooms_FN
