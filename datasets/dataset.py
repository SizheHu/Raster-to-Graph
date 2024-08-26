import copy
import json
import os
from collections import defaultdict
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from util.data_utils import l1_dist
from util.graph_utils import graph_to_tensor
from util.image_id_dict import d
from util.mean_std import mean, std
from util.semantics_dict import semantics_dict


class MyDataset(Dataset):
    def __init__(self, img_path, annot_path, extract_roi):
        self.img_path = img_path
        self.quadtree_path = '/'.join(img_path.split('/')[:-1]) + '/annot_npy'
        self.mode = img_path.split('/')[-1]

        # load annotation
        with open(annot_path, 'r') as f:
            dataset = json.load(f)
        # images
        self.imgs = {}
        for img in dataset['images']:
            self.imgs[img['id']] = img
        self.imgToAnns = defaultdict(list)
        for ann in dataset['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
        self.ids = list(sorted(self.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_file_name = self.imgs[img_id]['file_name']
        img = Image.open(os.path.join(self.img_path, img_file_name)).convert('RGB')

        if 1:
            # get structure annotations
            anns = self.imgToAnns[img_id]
            new_anns = []
            for ann in anns:
                new_ann = copy.deepcopy(ann)
                new_ann['point'] = [ann['point'][0], ann['point'][1]]
                new_anns.append(new_ann)
            target = {'image_id': img_id, 'annotations': new_anns}
            orig_quadtree = np.load(os.path.join(self.quadtree_path,
                                                 img_file_name[:-4] + '.npy'), allow_pickle=True).item()['quatree'][0]
            quadtree = {}
            for k, v in orig_quadtree.items():
                new_k = k
                new_v = []
                for pos in v:
                    new_pos = (pos[0], pos[1])
                    new_v.append(new_pos)
                quadtree[new_k] = new_v

            orig_graph = np.load(os.path.join(self.quadtree_path,
                                              img_file_name[:-4] + '.npy'), allow_pickle=True).item()
            del orig_graph['quatree']
            new_graph = {}
            for k, v in orig_graph.items():
                new_k = (k[0], k[1])
                new_v = []
                for adj in v:
                    if adj == (-1, -1):
                        new_v.append((-1, -1))
                    else:
                        new_v.append((adj[0], adj[1]))
                new_graph[new_k] = new_v

            target_layers = []
            for layer, layer_points in quadtree.items():
                target_layer = []
                for layer_point in layer_points:
                    for target_i in target['annotations']:
                        if l1_dist(target_i['point'], list(layer_point)) <= 2:
                            target_layer.append(target_i)
                            break
                target_layers.extend(target_layer)
            layer_indices = []
            count = 0
            for k, v in quadtree.items():
                if k == 0:
                    layer_indices.append(0)
                else:
                    layer_indices.append(count)
                count += len(v)

            image_id = torch.tensor([d[img_id]])

            points = [obj['point'] for obj in target_layers]
            points = torch.as_tensor(points, dtype=torch.int64).reshape(-1, 2)
            edges = [obj['edge_code'] for obj in target_layers]
            edges = torch.tensor(edges, dtype=torch.int64)

            # get semantic annotations
            semantic_left_up = [semantics_dict[obj['semantic'][0]] for obj in target_layers]
            semantic_right_up = [semantics_dict[obj['semantic'][1]] for obj in target_layers]
            semantic_right_down = [semantics_dict[obj['semantic'][2]] for obj in target_layers]
            semantic_left_down = [semantics_dict[obj['semantic'][3]] for obj in target_layers]
            semantic_left_up = torch.tensor(semantic_left_up, dtype=torch.int64)
            semantic_right_up = torch.tensor(semantic_right_up, dtype=torch.int64)
            semantic_right_down = torch.tensor(semantic_right_down, dtype=torch.int64)
            semantic_left_down = torch.tensor(semantic_left_down, dtype=torch.int64)


            # annotations
            target = {}
            target["edges"] = edges
            target["image_id"] = image_id
            target["size"] = torch.as_tensor([img.size[1], img.size[0]])

            target["semantic_left_up"] = semantic_left_up
            target["semantic_right_up"] = semantic_right_up
            target["semantic_right_down"] = semantic_right_down
            target["semantic_left_down"] = semantic_left_down

            # get image
            img = F.to_tensor(img)
            img = F.normalize(img, mean=mean, std=std)
            target['unnormalized_points'] = points
            # normalize
            points = points / torch.tensor([img.shape[2], img.shape[1]], dtype=torch.float32)
            target["points"] = points
            target['layer_indices'] = torch.tensor(layer_indices)

            target['graph'] = graph_to_tensor(new_graph)

            return img, target


    def __len__(self):
        return len(self.ids)
