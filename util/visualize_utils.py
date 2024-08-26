import copy

import cv2
import numpy as np

from util.edges_utils import get_edges_alldirections
from util.metric_utils import get_results, get_results_visual
from util.semantics_dict import semantics_dict, semantics_dict_rev, semantics_dict_color



def visualize_simplenet_singlelayer(unnormalized, output_points, result_edges_confidence, epoch, output_dir, index):
    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    if output_points is not None:
        output_points = output_points.cpu().numpy().astype(np.int)
        for i, output_point in enumerate(output_points):
            cv2.rectangle(unnormalized,
                          (output_point[0] - 5, output_point[1] - 5),
                          (output_point[0] + 5, output_point[1] + 5),
                          color=(0, 0, 255),
                          thickness=-1)
            if result_edges_confidence is not None:
                edges = get_edges_alldirections(result_edges_confidence[i].item())
                up_edge = int(edges[0])
                left_edge = int(edges[1])
                down_edge = int(edges[2])
                right_edge = int(edges[3])
                if up_edge:
                    cv2.line(unnormalized,
                             (output_point[0], output_point[1]),
                             (output_point[0], output_point[1] - 12),
                             color=(0, 0, 255),
                             thickness=2)
                if left_edge:
                    cv2.line(unnormalized,
                             (output_point[0], output_point[1]),
                             (output_point[0] - 12, output_point[1]),
                             color=(0, 0, 255),
                             thickness=2)
                if down_edge:
                    cv2.line(unnormalized,
                             (output_point[0], output_point[1]),
                             (output_point[0], output_point[1] + 12),
                             color=(0, 0, 255),
                             thickness=2)
                if right_edge:
                    cv2.line(unnormalized,
                             (output_point[0], output_point[1]),
                             (output_point[0] + 12, output_point[1]),
                             color=(0, 0, 255),
                             thickness=2)

    cv2.imwrite(output_dir + '/val_visualize' + '/epoch' + str(epoch) + '/' + str(index) + '.jpg', unnormalized)



def visualize_simplenet_singlelayer101(unnormalized, output_points,
                                       result_edges_confidence, result_last_edges_confidence, result_this_edges_confidence,
                                       result_semantic_left_up_confidence, result_semantic_right_up_confidence,
                                       result_semantic_right_down_confidence, result_semantic_left_down_confidence,
                                       epoch, output_dir, index):

    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    if output_points is not None:
        output_points = output_points.cpu().numpy().astype(np.int)
        for i, output_point in enumerate(output_points):
            cv2.rectangle(unnormalized,
                          (output_point[0] - 5, output_point[1] - 5),
                          (output_point[0] + 5, output_point[1] + 5),
                          color=(0, 0, 255),
                          thickness=-1)
            if result_edges_confidence is not None and result_last_edges_confidence is not None and result_this_edges_confidence is not None:
                if result_edges_confidence[i].item() == 16 or result_last_edges_confidence[i].item() == 16 or result_this_edges_confidence[i].item() == 16:
                    pass
                else:
                    edges = get_edges_alldirections(result_edges_confidence[i].item())
                    last_edges = get_edges_alldirections(result_last_edges_confidence[i].item())
                    this_edges = get_edges_alldirections(result_this_edges_confidence[i].item())

                    up_edge = int(edges[0])
                    left_edge = int(edges[1])
                    down_edge = int(edges[2])
                    right_edge = int(edges[3])
                    if up_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] - 20),
                                 color=(0, 0, 255),
                                 thickness=2)
                    if left_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] - 20, output_point[1]),
                                 color=(0, 0, 255),
                                 thickness=2)
                    if down_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] + 20),
                                 color=(0, 0, 255),
                                 thickness=2)
                    if right_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] + 20, output_point[1]),
                                 color=(0, 0, 255),
                                 thickness=2)

                    up_last_edge = int(last_edges[0])
                    left_last_edge = int(last_edges[1])
                    down_last_edge = int(last_edges[2])
                    right_last_edge = int(last_edges[3])
                    if up_last_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] - 12),
                                 color=(0, 255, 0),
                                 thickness=2)
                    if left_last_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] - 12, output_point[1]),
                                 color=(0, 255, 0),
                                 thickness=2)
                    if down_last_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] + 12),
                                 color=(0, 255, 0),
                                 thickness=2)
                    if right_last_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] + 12, output_point[1]),
                                 color=(0, 255, 0),
                                 thickness=2)

                    up_this_edge = int(this_edges[0])
                    left_this_edge = int(this_edges[1])
                    down_this_edge = int(this_edges[2])
                    right_this_edge = int(this_edges[3])
                    if up_this_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] - 12),
                                 color=(255, 0, 255),
                                 thickness=2)
                    if left_this_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] - 12, output_point[1]),
                                 color=(255, 0, 255),
                                 thickness=2)
                    if down_this_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0], output_point[1] + 12),
                                 color=(255, 0, 255),
                                 thickness=2)
                    if right_this_edge:
                        cv2.line(unnormalized,
                                 (output_point[0], output_point[1]),
                                 (output_point[0] + 12, output_point[1]),
                                 color=(255, 0, 255),
                                 thickness=2)

            if result_semantic_left_up_confidence is not None and result_semantic_right_up_confidence is not None \
                    and result_semantic_right_down_confidence is not None and result_semantic_left_down_confidence is not None:
                if result_edges_confidence[i].item() == 16:
                    pass
                else:
                    semantic_left_up = semantics_dict_rev[result_semantic_left_up_confidence[i].item()]
                    semantic_right_up = semantics_dict_rev[result_semantic_right_up_confidence[i].item()]
                    semantic_right_down = semantics_dict_rev[result_semantic_right_down_confidence[i].item()]
                    semantic_left_down = semantics_dict_rev[result_semantic_left_down_confidence[i].item()]
                    cv2.putText(unnormalized, semantic_left_up, (output_point[0] - 22, output_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                semantics_dict_color[semantic_left_up], 1)
                    cv2.putText(unnormalized, semantic_right_up, (output_point[0] + 5, output_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                semantics_dict_color[semantic_right_up], 1)
                    cv2.putText(unnormalized, semantic_right_down, (output_point[0] + 5, output_point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                semantics_dict_color[semantic_right_down], 1)
                    cv2.putText(unnormalized, semantic_left_down, (output_point[0] - 22, output_point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                semantics_dict_color[semantic_left_down], 1)


    cv2.imwrite(output_dir + '/val_visualize' + '/epoch' + str(epoch) + '/' + str(index) + '.jpg', unnormalized)

    

semantics_dict_color_visual = {'living_room': (193, 255, 225),
              'kitchen': (203,204,254),
              'bedroom': (143,246,255),
              'bathroom': (255,255,191),
              'restroom': (240,255,240),
              'balcony': (106,181,249),
              'closet': (159,121,238),
              'corridor': (154,213,232),
              'washing_room': (254,232,160),
              'PS': (87,139,46),
              'outside': (255,255,255),
              'wall': (0,0,0),
              'no_type': (190,190,190)}

def visualize_monte(unnormalized, best_result, epoch, output_dir, index, d_rev, simple_cycles, results):
    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    unnormalized[499:511, 499:511, :] = 255

    aa_scale = 8
  

    unnormalized2 = copy.deepcopy(unnormalized)
    unnormalized2 = np.ascontiguousarray(unnormalized2)
    unnormalized2[:, :, :] = 255
    
    unnormalized2 = cv2.resize(unnormalized2, (aa_scale * 512, aa_scale * 512))
    
    unnormalized3 = copy.deepcopy(unnormalized)
    unnormalized3 = np.ascontiguousarray(unnormalized3)
    unnormalized3[:, :, :] = 255
    
    unnormalized3 = cv2.resize(unnormalized3, (aa_scale * 512, aa_scale * 512))
    
    unnormalized4 = copy.deepcopy(unnormalized)
    unnormalized4 = np.ascontiguousarray(unnormalized4)
    unnormalized4[:, :, :] = 255
    
    unnormalized4 = cv2.resize(unnormalized4, (aa_scale * 512, aa_scale * 512))

    if best_result[1] > 0:
        output_points, output_edges = get_results(best_result)
        output_points_li, output_edges_li, lcounts = get_results_visual(best_result) # 分层的

        output_points = [(aa_scale * p[0], aa_scale * p[1]) for p in output_points]
        output_edges = [((aa_scale * e[0][0], aa_scale * e[0][1]), (aa_scale * e[1][0], aa_scale * e[1][1])) for e in output_edges]
        output_points_li = [[pi[0], (aa_scale * pi[1][0], aa_scale * pi[1][1])] for pi in output_points_li]
        output_edges_li = [[ei[0], ((aa_scale * ei[1][0][0], aa_scale * ei[1][0][1]), (aa_scale * ei[1][1][0], aa_scale * ei[1][1][1]))] for ei in output_edges_li]
        

        for cycle_ind, simple_cycle in enumerate(simple_cycles):
            semantic_result = results[cycle_ind]
            polygon = [list((int(point_ind[0] * aa_scale), int(point_ind[1]  * aa_scale))) for point_ind in simple_cycle]
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            # cv2.polylines(unnormalized2, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.fillPoly(unnormalized2, [pts], color=semantics_dict_color_visual[semantics_dict_rev[semantic_result]])
        

        for output_edge in output_edges:
            cv2.line(unnormalized2, output_edge[0], output_edge[1], color=(0, 0, 0), thickness=3 * aa_scale, lineType=cv2.LINE_4)
        for output_edge in output_edges:
            cv2.line(unnormalized3, output_edge[0], output_edge[1], color=(0, 0, 0), thickness=3 * aa_scale, lineType=cv2.LINE_4)

        for output_point in output_points:
            cv2.circle(unnormalized3,
                          (output_point[0], output_point[1]),
                          color=(0, 215, 255),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)

        for i in range(lcounts):
            sss = []
            unnormalized4_copy = copy.deepcopy(unnormalized4)
            unnormalized4_copy2 = copy.deepcopy(unnormalized4)
            output_edges_i = [(t[0], t[1]) for t in output_edges_li if t[0] == i]
            output_points_i = [(t[0], t[1]) for t in output_points_li if t[0] == i]
            output_points_all = [(t[0], t[1]) for t in output_points_li]
            

            for output_edge_i in output_edges_i:
                # cv2.line(unnormalized4, output_edge_i[0], output_edge_i[1], color=(0, 255, 0), thickness=2 * aa_scale, lineType=cv2.LINE_AA)
                cv2.line(unnormalized4, output_edge_i[1][0], output_edge_i[1][1], color=(0, 0, 0), thickness=3 * aa_scale, lineType=cv2.LINE_AA)
                p1 = output_edge_i[1][0]
                p2 = output_edge_i[1][1]
                p1_l = -1
                p2_l = -1
                for p in output_points_all:
                    if abs(p[1][0] - p1[0]) + abs(p[1][1] - p1[1]) <= 2:
                        p1_l = p[0]
                    elif abs(p[1][0] - p2[0]) + abs(p[1][1] - p2[1]) <= 2:
                        p2_l = p[0]
                assert (p1_l != -1 and p2_l != -1)
                if p1_l != p2_l:
                    cv2.line(unnormalized4_copy, output_edge_i[1][0], output_edge_i[1][1], color=(0, 255, 0), thickness=3 * aa_scale, lineType=cv2.LINE_AA)
                else:
                    cv2.line(unnormalized4_copy, output_edge_i[1][0], output_edge_i[1][1], color=(0, 0, 255), thickness=3 * aa_scale, lineType=cv2.LINE_AA)
            for output_point_i in output_points_i:
                sss.append((output_point_i[1][0], output_point_i[1][1]))
            # print(i, sss)
            for oi in sss:
                # cv2.rectangle(unnormalized4,
                #               (output_point_i[1][0] - 5, output_point_i[1][1] - 5),
                #               (output_point_i[1][0] + 5, output_point_i[1][1] + 5),
                #               color=(0,0,255),
                #               thickness=-1)
                cv2.circle(unnormalized4,
                          oi,
                          color=(0, 0, 0),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy,
                          oi,
                          color=(0, 215, 255),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy2,
                          oi,
                          color=(0, 215, 255),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
            for oilast in [t[1] for t in output_points_li if t[0] == i-1]:
                cv2.circle(unnormalized4_copy,
                          oilast,
                          color=(0, 0, 0),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy2,
                          oilast,
                          color=(0, 0, 0),
                          radius=4 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
            unnormalized4_copy = cv2.resize(unnormalized4_copy, (512, 512), interpolation=cv2.INTER_LINEAR)
            unnormalized4_copy2 = cv2.resize(unnormalized4_copy2, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            
            cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_vis.jpg', unnormalized4_copy)
            cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_vis_noedge.jpg', unnormalized4_copy2)


    
    
    unnormalized2 = cv2.resize(unnormalized2, (512, 512), interpolation=cv2.INTER_LINEAR)
    unnormalized3 = cv2.resize(unnormalized3, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    
    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '.jpg', unnormalized)
    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '_semantics' + '.jpg', unnormalized2)
    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '_structure' + '.jpg', unnormalized3)



def visualize_monte_for_overview(unnormalized, best_result, epoch, output_dir, index, d_rev, simple_cycles, results):
    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    unnormalized[499:511, 499:511, :] = 255
    aa_scale = 8
    
    unnormalized3 = copy.deepcopy(unnormalized)
    unnormalized3 = np.ascontiguousarray(unnormalized3)
    unnormalized3[:, :, :] = 255
    
    unnormalized3 = cv2.resize(unnormalized3, (aa_scale * 512, aa_scale * 512))
    
    unnormalized4 = cv2.addWeighted(cv2.resize(unnormalized, (aa_scale * 512, aa_scale * 512)), 1, np.ascontiguousarray(np.ones((512 * aa_scale, 512 * aa_scale, 3), dtype=unnormalized.dtype) * 255), 0, 0) # 实图

    if best_result[1] > 0:
        output_points, output_edges = get_results(best_result)
        output_points_li, output_edges_li, lcounts = get_results_visual(best_result)

        output_points = [(aa_scale * p[0], aa_scale * p[1]) for p in output_points]
        output_edges = [((aa_scale * e[0][0], aa_scale * e[0][1]), (aa_scale * e[1][0], aa_scale * e[1][1])) for e in output_edges]
        output_points_li = [[pi[0], (aa_scale * pi[1][0], aa_scale * pi[1][1])] for pi in output_points_li]
        output_edges_li = [[ei[0], ((aa_scale * ei[1][0][0], aa_scale * ei[1][0][1]), (aa_scale * ei[1][1][0], aa_scale * ei[1][1][1]))] for ei in output_edges_li]

        for cycle_ind, simple_cycle in enumerate(simple_cycles):
            semantic_result = results[cycle_ind]
            polygon = [list((int(point_ind[0] * aa_scale), int(point_ind[1]  * aa_scale))) for point_ind in simple_cycle]
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(unnormalized3, [pts], color=semantics_dict_color_visual[semantics_dict_rev[semantic_result]])

        for output_edge in output_edges:
            cv2.line(unnormalized3, output_edge[0], output_edge[1], color=(255, 0, 0), thickness=5 * aa_scale, lineType=cv2.LINE_4)

        for output_point in output_points:
            cv2.circle(unnormalized3,
                          (output_point[0], output_point[1]),
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
        sss = []
        output_points_li = [(0, (512, 1152)),
                            (1, (760, 1152)),
                            (2, (512, 2928)),
                            (2, (1272, 1152)),
                            (3, (1272, 1408)),]

        output_edges_li_temp = []
        for t in output_edges_li:
            t0 = -1
            t1 = -1
            for x in output_points_li:
                if x[1] == t[1][0]:
                    t0 = x[0]
                elif x[1] == t[1][1]:
                    t1 = x[0]
            # assert t0 != -1 and t1 != -1
            if t0 != -1 and t1 != -1:
                output_edges_li_temp.append((max(t0, t1), (t[1][0], t[1][1])))
        output_edges_li = output_edges_li_temp
        # print(output_edges_li)
        # assert 0
        lcounts = 4

        

        for i in range(lcounts):
            sss = []
            unnormalized4_copy = copy.deepcopy(unnormalized4)
            unnormalized4_copy2 = copy.deepcopy(unnormalized4)
            output_edges_i = [(t[0], t[1]) for t in output_edges_li if t[0] == i]
            output_points_i = [(t[0], t[1]) for t in output_points_li if t[0] == i]
            output_points_all = [(t[0], t[1]) for t in output_points_li]
            


            for output_edge_i in output_edges_i:
                cv2.line(unnormalized4, output_edge_i[1][0], output_edge_i[1][1], color=(255, 0, 0), thickness=5 * aa_scale, lineType=cv2.LINE_AA)
                p1 = output_edge_i[1][0]
                p2 = output_edge_i[1][1]
                p1_l = -1
                p2_l = -1
                for p in output_points_all:
                    if abs(p[1][0] - p1[0]) + abs(p[1][1] - p1[1]) <= 2:
                        p1_l = p[0]
                    elif abs(p[1][0] - p2[0]) + abs(p[1][1] - p2[1]) <= 2:
                        p2_l = p[0]
                assert (p1_l != -1 and p2_l != -1)
                if p1_l != p2_l:
                    cv2.line(unnormalized4_copy, output_edge_i[1][0], output_edge_i[1][1], color=(255, 0, 0), thickness=5 * aa_scale, lineType=cv2.LINE_AA)
                else:
                    cv2.line(unnormalized4_copy, output_edge_i[1][0], output_edge_i[1][1], color=(255, 0, 0), thickness=5 * aa_scale, lineType=cv2.LINE_AA)
            for output_point_i in output_points_i:
                sss.append((output_point_i[1][0], output_point_i[1][1]))
            # print(i, sss)
            unnormalized4_copy3 = copy.deepcopy(unnormalized4)
            for oi in sss:
                cv2.circle(unnormalized4_copy3,
                          oi,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
            unnormalized4_copy3 = cv2.resize(unnormalized4_copy3, (512, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_vis3.jpg', unnormalized4_copy3)
            for oi in sss:
                cv2.circle(unnormalized4,
                          oi,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy,
                          oi,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy2,
                          oi,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
            for oilast in [t[1] for t in output_points_li if t[0] <= i-1]:
                cv2.circle(unnormalized4_copy,
                          oilast,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(unnormalized4_copy2,
                          oilast,
                          color=(0, 215, 255),
                          radius=7 * aa_scale, thickness=-1, lineType=cv2.LINE_AA)
            unnormalized4_copy = cv2.resize(unnormalized4_copy, (512, 512), interpolation=cv2.INTER_LINEAR)
            unnormalized4_copy2 = cv2.resize(unnormalized4_copy2, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            
            # cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_itr.jpg', unnormalized4_copy)
            
            cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_vis.jpg', unnormalized4_copy)
            cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '-' + str(i) + '_vis_noedge.jpg', unnormalized4_copy2)
        

    unnormalized3 = cv2.resize(unnormalized3, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    
    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '.jpg', unnormalized)
    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '_seman' + '.jpg', unnormalized3)

def rasterize(unnormalized, best_result, epoch, output_dir, index, d_rev, simple_cycles, results):
    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    unnormalized[499:511, 499:511, :] = 255

    aa_scale = 1
    
    unnormalized3 = copy.deepcopy(unnormalized)
    unnormalized3 = np.ascontiguousarray(unnormalized3)
    unnormalized3[:, :, :] = 255

    if best_result[1] > 0:
        output_points, output_edges = get_results(best_result)
        for cycle_ind, simple_cycle in enumerate(simple_cycles):
            semantic_result = results[cycle_ind]
            polygon = [list((int(point_ind[0]), int(point_ind[1]))) for point_ind in simple_cycle]
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(unnormalized3, [pts], color=semantics_dict_color_visual[semantics_dict_rev[semantic_result]])
        for output_edge in output_edges:
            cv2.line(unnormalized3, output_edge[0], output_edge[1], color=(0, 0, 0), thickness=3, lineType=cv2.LINE_4)

    cv2.imwrite(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + '_seman' + '.png', unnormalized3)