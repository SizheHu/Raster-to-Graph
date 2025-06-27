import copy
import math
import random

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from util.geom_utils import is_clockwise_or_not, x_axis_angle, get_quadrant, counter_degree, rotate_degree_clockwise_from_counter_degree, \
    rotate_degree_counterclockwise_from_counter_degree, poly_area
from util.metric_utils import get_results, get_results_float_with_semantic


def graph_to_tensor(graph):
    t_l = []
    for k, v in graph.items():
        a = []
        a.append(k)
        a.extend(v)
        b = [list(i) for i in a]
        c = torch.tensor(b)
        t_l.append(c)
    return torch.stack(t_l, dim=0)

def tensor_to_graph(tensor):
    gr = {}
    for kv in tensor:
        k = tuple([i.item() for i in kv[0]])
        v = kv[1:5]
        v = v.tolist()
        v = [tuple(i) for i in v]
        gr[k] = v
    return gr

def tensors_to_graphs_batch(tensors):
    return [tensor_to_graph(ts) for ts in tensors]

def get_cycle_basis_and_semantic_deprecated(best_result):
    output_points, output_edges = get_results_float_with_semantic(best_result)
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))

    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])

    nx.draw(G)
    # plt.show()
    simple_cycles = nx.cycle_basis(G)



    results = []

    for cycle_ind, cycle in enumerate(simple_cycles):

        points = [d_rev[ind] for ind in cycle]
        points.append(points[0])

        is_clockwise = is_clockwise_or_not([(p[0], p[1]) for p in points])
        if is_clockwise:
            points.reverse()

        cross_products = []
        poses = [(p[0], p[1]) for p in points]
        for ind in range(len(poses) - 1):
            ei = [poses[(ind + 1) % (len(poses) - 1)][0] - poses[ind][0],
                  poses[(ind + 1) % (len(poses) - 1)][1] - poses[ind][1]]
            eiplus1 = [poses[(ind + 2) % (len(poses) - 1)][0] - poses[(ind + 1) % (len(poses) - 1)][0],
                  poses[(ind + 2) % (len(poses) - 1)][1] - poses[(ind + 1) % (len(poses) - 1)][1]]
            cross_products.append(np.cross(ei, eiplus1).tolist())
        cross_products.insert(0, cross_products[-1])
        cross_products.pop(-1)

        while 0 in cross_products:
            for point_ind, cross_product in enumerate(cross_products):
                if cross_product == 0:
                    if point_ind == 0:
                        p0 = copy.deepcopy(points[0])
                        points[0] = (p0[0] + 0.000001 * random.random() * [-1, 1][random.randint(0, 1)],
                                     p0[1] + 0.000001 * random.random() * [-1, 1][random.randint(0, 1)],
                                     p0[2], p0[3], p0[4], p0[5])
                        points[-1] = copy.deepcopy(points[0])
                    else:
                        pi = copy.deepcopy(points[point_ind])
                        points[point_ind] = (pi[0] + 0.000001 * random.random() * [-1, 1][random.randint(0, 1)],
                                     pi[1] + 0.000001 * random.random() * [-1, 1][random.randint(0, 1)],
                                     pi[2], pi[3], pi[4], pi[5])
            # print(points)
            cross_products = []
            poses = [(p[0], p[1]) for p in points]
            for ind in range(len(poses) - 1):
                ei = [poses[(ind + 1) % (len(poses) - 1)][0] - poses[ind][0],
                      poses[(ind + 1) % (len(poses) - 1)][1] - poses[ind][1]]
                eiplus1 = [poses[(ind + 2) % (len(poses) - 1)][0] - poses[(ind + 1) % (len(poses) - 1)][0],
                           poses[(ind + 2) % (len(poses) - 1)][1] - poses[(ind + 1) % (len(poses) - 1)][1]]
                cross_products.append(np.cross(ei, eiplus1).tolist())
            cross_products.insert(0, cross_products[-1])
            cross_products.pop(-1)

        semantics = [[p[2], p[3], p[4], p[5]] for p in points]


        degrees = []
        for ind in range(len(poses) - 1):
            ei_minus = [-(poses[(ind + 1) % (len(poses) - 1)][0] - poses[ind][0]),
                  -(poses[(ind + 1) % (len(poses) - 1)][1] - poses[ind][1])]

            eiplus1 = [poses[(ind + 2) % (len(poses) - 1)][0] - poses[(ind + 1) % (len(poses) - 1)][0],
                       poses[(ind + 2) % (len(poses) - 1)][1] - poses[(ind + 1) % (len(poses) - 1)][1]]

            degrees.append((x_axis_angle(ei_minus), x_axis_angle(eiplus1)))
        degrees.insert(0, degrees[-1])
        degrees.pop(-1)

        angles = []
        for degree in degrees:
            angles.append(((min(degree), max(degree)), (max(degree), min(degree))))

        angles_to_semantics = []
        for angle_ind, angle in enumerate(angles):
            angle1 = angle[0]
            angle2 = angle[1]
            quadrant1 = get_quadrant(angle1)
            quadrant2 = get_quadrant(angle2)

            semantic1 = (semantics[angle_ind][1] if quadrant1[0] >= 45 else -1,
                         semantics[angle_ind][0] if quadrant1[1] >= 45 else -1,
                         semantics[angle_ind][3] if quadrant1[2] >= 45 else -1,
                         semantics[angle_ind][2] if quadrant1[3] >= 45 else -1)
            semantic2 = (semantics[angle_ind][1] if quadrant2[0] >= 45 else -1,
                         semantics[angle_ind][0] if quadrant2[1] >= 45 else -1,
                         semantics[angle_ind][3] if quadrant2[2] >= 45 else -1,
                         semantics[angle_ind][2] if quadrant2[3] >= 45 else -1)

            angle1_degree = sum(quadrant1)
            angle2_degree = sum(quadrant2)

            xproduct = cross_products[angle_ind]

            if xproduct < 0:
                if angle1_degree < angle2_degree:
                    angles_to_semantics.append(semantic1)
                else:
                    angles_to_semantics.append(semantic2)
            elif xproduct > 0:
                if angle1_degree < angle2_degree:
                    angles_to_semantics.append(semantic2)
                else:
                    angles_to_semantics.append(semantic1)
            else:
                assert 0


        semantic_result = {}
        for semantic_label in range(0, 13):
            semantic_result[semantic_label] = 0
        for everypoint_semantic in angles_to_semantics:
            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
            for label in everypoint_semantic:
                semantic_result[label] += 1 / len(everypoint_semantic)



        this_cycle_semantic1 = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
        this_cycle_result = None
        if this_cycle_semantic1[0][1] > this_cycle_semantic1[1][1]:
            this_cycle_result = this_cycle_semantic1[0][0]
        else:
            this_cycle_results = [i[0] for i in this_cycle_semantic1 if i[1] == this_cycle_semantic1[0][1]]
            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
        results.append(this_cycle_result)

    return d_rev, simple_cycles, results


def get_cycle_basis_and_semantic(best_result):
    # The "best" in the name has nothing to do with our actual practice. It is just a legacy of our early attempts at Monte Carlo thinking.
    output_points, output_edges = get_results_float_with_semantic(best_result)
    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)


    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))
    # print(d)

    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])

    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    bridges = list(nx.bridges(G))
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    connected_components = list(nx.connected_components(G))
    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    simple_cycle_semantics = []
                    simple_cycle = []
                    simple_cycle_number = []
                    point1 = edge_c[0]
                    point2 = edge_c[1]
                    point1_number = d[point1]
                    point2_number = d[point2]

                    initial_point = None
                    initial_point_number = None
                    if point1_number < point2_number:
                        initial_point = point1
                        initial_point_number = point1_number
                    else:
                        initial_point = point2
                        initial_point_number = point2_number
                    simple_cycle.append(initial_point)
                    simple_cycle_number.append(initial_point_number)

                    last_point = initial_point
                    last_point_number = initial_point_number

                    current_point = None
                    current_point_number = None
                    if point1_number < point2_number:
                        current_point = point2
                        current_point_number = point2_number
                    else:
                        current_point = point1
                        current_point_number = point1_number
                    simple_cycle.append(current_point)
                    simple_cycle_number.append(current_point_number)

                    next_initial_point = copy.deepcopy(current_point)
                    next_initial_point_number = copy.deepcopy(current_point_number)

                    next_point = None
                    next_point_number = None

                    while next_point != next_initial_point:

                        relevant_edges = []
                        for edge in output_edges_c:
                            if edge[0] == current_point or edge[1] == current_point:
                                relevant_edges.append(edge)

                        relevant_edges_degree = []
                        for relevant_edge in relevant_edges:

                            vec = None
                            if relevant_edge[0] == current_point:
                                vec = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                            elif relevant_edge[1] == current_point:
                                vec = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                            else:
                                assert 0

                            vec_degree = x_axis_angle(vec)
                            relevant_edges_degree.append(vec_degree)

                        vec_from_current_point_to_last_point = None
                        vec_from_current_point_to_last_point_degree = None
                        for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                            if relevant_edge == (current_point, last_point):
                                vec_from_current_point_to_last_point = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            elif relevant_edge == (last_point, current_point):
                                vec_from_current_point_to_last_point = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            else:
                                continue

                        rotate_deltas_counterclockwise = []

                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)

                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))

                        interior_angle_counterclockwise = interior_angles[max_rotate_index]

                        current_point_semantic = [current_point[3], current_point[2], current_point[5], current_point[4]]

                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)
                        quadrant_smaller_to_bigger_counterclockwise = get_quadrant((interior_angle_counterclockwise_degree_smaller,
                                                                                    interior_angle_counterclockwise_degree_bigger))
                        # print(quadrant_smaller_to_bigger_counterclockwise)
                        if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                            pass
                        elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                            quadrant_smaller_to_bigger_counterclockwise = (90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[3])
                        else:
                            assert 0

                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if quadrant_smaller_to_bigger_counterclockwise[qd] >= 45:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)

                        simple_cycle_semantics.append(current_point_semantic_valid)


                        max_rotate_edge = relevant_edges[max_rotate_index]

                        if max_rotate_edge[0] == current_point:
                            next_point = max_rotate_edge[1]
                            next_point_number = d[next_point]
                        elif max_rotate_edge[1] == current_point:
                            next_point = max_rotate_edge[0]
                            next_point_number = d[next_point]
                        else:
                            assert 0

                        last_point = current_point
                        last_point_number = current_point_number
                        current_point = next_point
                        current_point_number = next_point_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)

                    for point_number_ind, point_number in enumerate(simple_cycle_number):
                        if point_number_ind < len(simple_cycle_number) - 1:
                            edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                            # print(simple_cycle_number)
                            if edge_number[0] < edge_number[1]:
                                if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                elif (d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[1]], d_rev[edge_number[0]]))

                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)

                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)

                        semantic_result = {}
                        for semantic_label in range(0, 13):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
                            for label in everypoint_semantic:
                                semantic_result[label] += 1 / len(everypoint_semantic)
                        # print(semantic_result)
                        del semantic_result[11]
                        del semantic_result[12]

                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)



    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics


def get_cycle_basis_and_semantic_2(best_result):
    output_points, output_edges = get_results_float_with_semantic(best_result)
    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)
    # print(output_points)
    # print(output_edges)
    # assert 0
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))
    # print(d)

    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])

    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []

    bridges = list(nx.bridges(G))

    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])

    connected_components = list(nx.connected_components(G))
    # print(connected_components)
    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c] # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c) # 用于遍历以减少时间复杂度的边集，其中的边会删除
            # print(output_points_c)
            # print(output_edges_c)

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    simple_cycle_semantics = []
                    simple_cycle = []
                    simple_cycle_number = []
                    point1 = edge_c[0]
                    point2 = edge_c[1]
                    point1_number = d[point1]
                    point2_number = d[point2]

                    initial_point = None
                    initial_point_number = None
                    if point1_number < point2_number:
                        initial_point = point1
                        initial_point_number = point1_number
                    else:
                        initial_point = point2
                        initial_point_number = point2_number
                    simple_cycle.append(initial_point)
                    simple_cycle_number.append(initial_point_number)

                    last_point = initial_point
                    last_point_number = initial_point_number

                    current_point = None
                    current_point_number = None
                    if point1_number < point2_number:
                        current_point = point2
                        current_point_number = point2_number
                    else:
                        current_point = point1
                        current_point_number = point1_number
                    simple_cycle.append(current_point)
                    simple_cycle_number.append(current_point_number)

                    next_initial_point = copy.deepcopy(current_point)
                    next_initial_point_number = copy.deepcopy(current_point_number)

                    next_point = None
                    next_point_number = None

                    while next_point != next_initial_point:

                        relevant_edges = []
                        for edge in output_edges_c:
                            if edge[0] == current_point or edge[1] == current_point:
                                relevant_edges.append(edge)

                        relevant_edges_degree = []
                        for relevant_edge in relevant_edges:

                            vec = None
                            if relevant_edge[0] == current_point:
                                vec = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                            elif relevant_edge[1] == current_point:
                                vec = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                            else:
                                assert 0

                            vec_degree = x_axis_angle(vec)
                            relevant_edges_degree.append(vec_degree)

                        vec_from_current_point_to_last_point = None
                        vec_from_current_point_to_last_point_degree = None
                        for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                            if relevant_edge == (current_point, last_point):
                                vec_from_current_point_to_last_point = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            elif relevant_edge == (last_point, current_point):
                                vec_from_current_point_to_last_point = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            else:
                                continue

                        rotate_deltas_counterclockwise = []
                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)
                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                        interior_angle_counterclockwise = interior_angles[max_rotate_index]
                        current_point_semantic = [current_point[3], current_point[2], current_point[5], current_point[4]]
                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)
                        quadrant_smaller_to_bigger_counterclockwise = get_quadrant((interior_angle_counterclockwise_degree_smaller,
                                                                                    interior_angle_counterclockwise_degree_bigger))
                        # print(quadrant_smaller_to_bigger_counterclockwise)
                        if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                            pass
                        elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                            quadrant_smaller_to_bigger_counterclockwise = (90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[3])
                        else:
                            assert 0
                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if 1:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)
                        simple_cycle_semantics.append(current_point_semantic_valid)

                        max_rotate_edge = relevant_edges[max_rotate_index]
                        if max_rotate_edge[0] == current_point:
                            next_point = max_rotate_edge[1]
                            next_point_number = d[next_point]
                        elif max_rotate_edge[1] == current_point:
                            next_point = max_rotate_edge[0]
                            next_point_number = d[next_point]
                        else:
                            assert 0

                        last_point = current_point
                        last_point_number = current_point_number
                        current_point = next_point
                        current_point_number = next_point_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)

                    for point_number_ind, point_number in enumerate(simple_cycle_number):
                        if point_number_ind < len(simple_cycle_number) - 1:
                            edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                            # print(simple_cycle_number)
                            if edge_number[0] < edge_number[1]:
                                if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                elif (d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[1]], d_rev[edge_number[0]]))

                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)
                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)
                        semantic_result = {}
                        for semantic_label in range(0, 13):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            for _ in range(0, 13):
                                if _ in everypoint_semantic:
                                    semantic_result[_] += 1
                        del semantic_result[11]
                        del semantic_result[12]

                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)



    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

