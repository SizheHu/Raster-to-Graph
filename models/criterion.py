import torch
import torch.nn.functional as F
from torch import nn
from util.nn_utils import sigmoid_focal_loss


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict


    def loss_edges(self, outputs, targets, indices, num_points):
        output_edges = outputs['pred_edges']

        target_edges = torch.zeros((output_edges.shape[0], output_edges.shape[1], output_edges.shape[2]),
                                     dtype=output_edges.dtype, device=output_edges.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['edges'][gt_index_i]
                target_edges[bs_i, obj_indices[index_i], gt] = 1

        loss_ce = sigmoid_focal_loss(output_edges, target_edges, num_points, alpha=0.25, gamma=2) * output_edges.shape[1] # * 500

        ''' '''
        return loss_ce

    def loss_last_edges(self, outputs, targets, indices, num_points):
        output_last_edges = outputs['pred_last_edges']

        target_last_edges = torch.zeros((output_last_edges.shape[0], output_last_edges.shape[1], output_last_edges.shape[2]),
                                        dtype=output_last_edges.dtype, device=output_last_edges.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['last_edges'][gt_index_i]
                target_last_edges[bs_i, obj_indices[index_i], gt] = 1

        loss_ce = sigmoid_focal_loss(output_last_edges, target_last_edges, num_points, alpha=0.25, gamma=2) * output_last_edges.shape[1]

        return loss_ce

    def loss_this_edges(self, outputs, targets, indices, num_points):
        output_this_edges = outputs['pred_this_edges']

        target_this_edges = torch.zeros((output_this_edges.shape[0], output_this_edges.shape[1], output_this_edges.shape[2]),
                                        dtype=output_this_edges.dtype, device=output_this_edges.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['this_edges'][gt_index_i]
                target_this_edges[bs_i, obj_indices[index_i], gt] = 1

        loss_ce = sigmoid_focal_loss(output_this_edges, target_this_edges, num_points, alpha=0.25, gamma=2) * output_this_edges.shape[1]

        return loss_ce


    def loss_semantic_left_up(self, outputs, targets, indices, num_points):
        output_semantic_left_up = outputs['pred_semantic_left_up']

        target_semantic_left_up = torch.zeros((output_semantic_left_up.shape[0], output_semantic_left_up.shape[1], output_semantic_left_up.shape[2]),
                                        dtype=output_semantic_left_up.dtype, device=output_semantic_left_up.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['semantic_left_up'][gt_index_i]
                target_semantic_left_up[bs_i, obj_indices[index_i], gt] = 1

        loss_semantic_left_up = sigmoid_focal_loss(output_semantic_left_up, target_semantic_left_up, num_points, alpha=0.25, gamma=2) * output_semantic_left_up.shape[1]

        return loss_semantic_left_up

    def loss_semantic_right_up(self, outputs, targets, indices, num_points):
        output_semantic_right_up = outputs['pred_semantic_right_up']

        target_semantic_right_up = torch.zeros(
            (output_semantic_right_up.shape[0], output_semantic_right_up.shape[1], output_semantic_right_up.shape[2]),
            dtype=output_semantic_right_up.dtype, device=output_semantic_right_up.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['semantic_right_up'][gt_index_i]
                target_semantic_right_up[bs_i, obj_indices[index_i], gt] = 1

        loss_semantic_right_up = sigmoid_focal_loss(output_semantic_right_up, target_semantic_right_up, num_points, alpha=0.25, gamma=2) * \
                                 output_semantic_right_up.shape[1]
        return loss_semantic_right_up

    def loss_semantic_right_down(self, outputs, targets, indices, num_points):
        output_semantic_right_down = outputs['pred_semantic_right_down']

        target_semantic_right_down = torch.zeros(
            (output_semantic_right_down.shape[0], output_semantic_right_down.shape[1], output_semantic_right_down.shape[2]),
            dtype=output_semantic_right_down.dtype, device=output_semantic_right_down.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['semantic_right_down'][gt_index_i]
                target_semantic_right_down[bs_i, obj_indices[index_i], gt] = 1

        loss_semantic_right_down = sigmoid_focal_loss(output_semantic_right_down, target_semantic_right_down, num_points, alpha=0.25, gamma=2) * \
                                   output_semantic_right_down.shape[1]
        return loss_semantic_right_down

    def loss_semantic_left_down(self, outputs, targets, indices, num_points):
        output_semantic_left_down = outputs['pred_semantic_left_down']

        target_semantic_left_down = torch.zeros(
            (output_semantic_left_down.shape[0], output_semantic_left_down.shape[1], output_semantic_left_down.shape[2]),
            dtype=output_semantic_left_down.dtype, device=output_semantic_left_down.device)

        for bs_i, hungarian_indices in enumerate(indices):
            obj_indices = hungarian_indices[0]
            gt_indices = hungarian_indices[1]
            for index_i in range(len(obj_indices)):
                gt_index_i = gt_indices[index_i]
                gt = targets[bs_i]['semantic_left_down'][gt_index_i]
                target_semantic_left_down[bs_i, obj_indices[index_i], gt] = 1

        loss_semantic_left_down = sigmoid_focal_loss(output_semantic_left_down, target_semantic_left_down, num_points, alpha=0.25, gamma=2) * \
                                  output_semantic_left_down.shape[1]
        return loss_semantic_left_down


    def loss_points(self, outputs, targets, indices, num_points):
        idx = self._get_src_permutation_idx(indices)
        # print(idx) # (tensor([0, 0, 1, 1, 1, 1, 1]) -> batch, tensor([206, 433,  38,  78, 157, 197, 413]))
        src_points = outputs['pred_points'][idx]
        # print(src_points.shape) # torch.Size([7, 2])
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # print(target_points.shape) # torch.Size([7, 2])
        loss_point = F.l1_loss(src_points, target_points, reduction='none')
        # print(target_points.shape) # torch.Size([7, 2])
        loss_point = loss_point.sum() / num_points
        # print('loss_point', loss_point)
        # assert 0
        return loss_point

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        num_points = sum(len(t["edges"]) for t in targets)

        return {'loss_point': self.loss_points(outputs, targets, indices, num_points),
                'loss_edge': self.loss_edges(outputs, targets, indices, num_points),
                'loss_last_edge': self.loss_last_edges(outputs, targets, indices, num_points),
                'loss_this_edge': self.loss_this_edges(outputs, targets, indices, num_points),
                'loss_semantic_left_up': self.loss_semantic_left_up(outputs, targets, indices, num_points),
                'loss_semantic_right_up': self.loss_semantic_right_up(outputs, targets, indices, num_points),
                'loss_semantic_right_down': self.loss_semantic_right_down(outputs, targets, indices, num_points),
                'loss_semantic_left_down': self.loss_semantic_left_down(outputs, targets, indices, num_points)}