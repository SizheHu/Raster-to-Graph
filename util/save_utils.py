import os
import copy

import cv2
import numpy as np
import random

from util.edges_utils import get_edges_alldirections
from util.metric_utils import get_results, get_results_visual
from util.semantics_dict import semantics_dict, semantics_dict_rev, semantics_dict_color


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
semantics_dict_color_visual_rev = dict(zip(semantics_dict_color_visual.values(), semantics_dict_color_visual.keys()))

def save_for_3djianmo(unnormalized, best_result, epoch, output_dir, index, d_rev, simple_cycles, results):
    unnormalized = unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]
    unnormalized = np.ascontiguousarray(unnormalized)
    unnormalized[499:511, 499:511, :] = 255

    aa_scale = 1
    if aa_scale == 8:
        mingming = '_vector'
    elif aa_scale == 1:
        mingming = '_vector_original'
    else:
        mingming = '_vector' + str(aa_scale)
  

    unnormalized2 = copy.deepcopy(unnormalized)
    unnormalized2 = np.ascontiguousarray(unnormalized2)
    unnormalized2[:, :, :] = 255
    
    unnormalized2 = cv2.resize(unnormalized2, (aa_scale * 512, aa_scale * 512))
    
    unnormalized3 = copy.deepcopy(unnormalized)
    unnormalized3 = np.ascontiguousarray(unnormalized3)
    unnormalized3[:, :, :] = 255
    
    unnormalized3 = cv2.resize(unnormalized3, (aa_scale * 512, aa_scale * 512))


    if best_result[1] > 0:
        semans = {}
        output_points, output_edges = get_results(best_result)

        output_points = [(aa_scale * p[0], aa_scale * p[1]) for p in output_points]
        output_edges = [((aa_scale * e[0][0], aa_scale * e[0][1]), (aa_scale * e[1][0], aa_scale * e[1][1])) for e in output_edges]
        
        f = open(output_dir + '/val_visualize_iter' + '/epoch' + str(epoch) + '/' + str(index) + mingming + '.txt', mode='w')
        for oe in output_edges:
            f.write(str(oe[0][0]) + '\t' + str(oe[0][1]) + '\t' + str(oe[1][0]) + '\t' + str(oe[1][1]) + '\t' + 'wall')
            f.write('\n')
        for cycle_ind, simple_cycle in enumerate(simple_cycles):
            semantic_result = results[cycle_ind]
            polygon = [list((int(point_ind[0] * aa_scale), int(point_ind[1]  * aa_scale))) for point_ind in simple_cycle]
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            # cv2.polylines(unnormalized2, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.fillPoly(unnormalized2, [pts], color=semantics_dict_color_visual[semantics_dict_rev[semantic_result]])
        for output_edge in output_edges:
            cv2.line(unnormalized2, output_edge[0], output_edge[1], color=(0, 0, 0), thickness=3 * aa_scale, lineType=cv2.LINE_8)
        gray = cv2.cvtColor(unnormalized2, cv2.COLOR_BGR2GRAY)
        ret, binimg = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        binimg = binimg.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binimg, connectivity=8)

        for i in range(len(stats)):
            if i >= 2:
                semans[i] = []
        stats = stats.tolist()
        centroids = centroids.tolist()
        for i in range(len(stats)):
            if i >= 2:
                randpoint1 = (random.randint(stats[i][0], stats[i][0] + stats[i][2]), random.randint(stats[i][1], stats[i][1] + stats[i][3]))
                randpoint2 = (random.randint(stats[i][0], stats[i][0] + stats[i][2]), random.randint(stats[i][1], stats[i][1] + stats[i][3]))
                count = 0
                while (not ((labels[randpoint1[1] - 3*aa_scale:randpoint1[1] + 3*aa_scale, randpoint1[0] - 3*aa_scale:randpoint1[0] + 3*aa_scale]) == i).all()) and count <= 100000:
                    randpoint1 = (random.randint(stats[i][0], stats[i][0] + stats[i][2]), random.randint(stats[i][1], stats[i][1] + stats[i][3]))
                    count += 1
                if count > 100000:
                    randpoint1 = (int(centroids[i][0]) - 1, int(centroids[i][1]) - 1)
                count = 0
                while (not ((labels[randpoint2[1] - 3*aa_scale:randpoint2[1] + 3*aa_scale, randpoint2[0] - 3*aa_scale:randpoint2[0] + 3*aa_scale]) == i).all()) and count <= 100000:
                    randpoint2 = (random.randint(stats[i][0], stats[i][0] + stats[i][2]), random.randint(stats[i][1], stats[i][1] + stats[i][3]))
                    count += 1
                if count > 100000:
                    randpoint2 = (int(centroids[i][0]) + 1, int(centroids[i][1]) + 1)
                if labels[randpoint1[1], randpoint1[0]] == i:
                    label = semantics_dict_color_visual_rev[tuple(unnormalized2[randpoint1[1], randpoint1[0], :].astype(np.uint8).tolist())]
                semans[i] = [randpoint1[0], randpoint1[1], randpoint2[0], randpoint2[1], label]
        for lbl, seman in semans.items():
            f.write(str(seman[0]) + '\t' + str(seman[1]) + '\t' + str(seman[2]) + '\t' + str(seman[3]) + '\t' + seman[4])
            f.write('\n')
                


        f.close()