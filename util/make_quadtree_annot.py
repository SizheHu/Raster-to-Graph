import copy
import os, numpy as np, collections, random
import time

import cv2




def get_key_by_value_first(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return None


def clockwise_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return 360 - theta * 180 / np.pi


def adj_direction(k, adj):
    cl_ang = clockwise_angle(np.array(k) - np.array(adj), np.array([0, 1]))
    if cl_ang < 45 or cl_ang > 315:
        return 'up'
    elif 45 < cl_ang < 135:
        return 'right'
    elif 135 < cl_ang < 225:
        return 'down'
    elif 225 < cl_ang < 315:
        return 'left'
    else:
        print(k, adj, cl_ang, np.sqrt((np.array(k)[0] - np.array(adj)[0]) ** 2 + (np.array(k)[1] - np.array(adj)[1]) ** 2))


def sort_corners(corners):
    return corners[np.lexsort([corners[:, 0], corners[:, 1]])]


def random_walk(dict, point):
    assert point in dict.keys()
    rand_seq = []
    now = point
    rand_seq.append(now)
    for _ in range(999999):
        adjs = list(set(dict[now]).difference({None}))
        rand_num = random.randint(0, 1000000) % len(adjs)
        now = adjs[rand_num]
        rand_seq.append(now)
    return rand_seq


def main():
    for fname in os.listdir(r'/data/r2v_2717/annot'):
        print(fname)
        dict = np.load(os.path.join(r'/data/r2v_2717/annot', fname),
                       allow_pickle=True).item()
        del dict['quatree']
        del dict['possort']
        # print(dict)
        dict_new = {}
        for k_temp, v_temp in dict.items():
            new_v_temp = []
            for v_i in v_temp:
                if v_i != (-1, -1):
                    new_v_temp.append(v_i)
                else:
                    new_v_temp.append(None)
            # print(new_v_temp)
            dict_new[k_temp] = new_v_temp
        # print(dict_new)


        quatree_list = []
        quatree_list_final = {}

        corners = None
        for k, v in dict_new.items():
            if corners is None:
                corners = np.array([k])
            else:
                corners = np.concatenate((corners, [k]), axis=0)
        root = sort_corners(corners)[0]

        l0 = []
        l0.append(tuple(root.tolist()))
        quatree_list.extend(l0)
        quatree_list_final[0] = l0


        quatree_append_li = []
        liminus1 = l0
        i = 1
        while len(quatree_append_li) > 0 or i == 1:
            quatree_append_li = []
            li = []
            for nodeiminus1 in liminus1:
                if nodeiminus1 is not None:
                    nodeiminus1_adj = dict_new[nodeiminus1]
                    li.extend(nodeiminus1_adj)
                else:
                    # li.extend([None, None, None, None])
                    pass
            for ni in li:
                if (ni not in quatree_list) and (ni is not None) and (ni not in quatree_append_li):
                    quatree_append_li.append(ni)
            quatree_list.extend(quatree_append_li)
            if len(quatree_append_li) > 0:
                quatree_list_final[i] = quatree_append_li


            liminus1 = li
            i += 1


        dict['quatree'] = [quatree_list_final]
        # print(dict)

        np.save(os.path.join(r'D:\PythonProjects\Deformable-DETR-main\data\r2v_2717\annot2', fname), dict)


if __name__ == '__main__':
    main()