import torch
from torch import nn


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        points, out_edges, out_last_edges, out_this_edges = \
            outputs['pred_points'], outputs['pred_edges'], outputs['pred_last_edges'], outputs['pred_this_edges']
        edge_probability = out_edges.sigmoid()
        last_edge_probability = out_last_edges.sigmoid()
        this_edge_probability = out_this_edges.sigmoid()

        out_semantic_left_up, out_semantic_right_up, out_semantic_right_down, out_semantic_left_down = \
            outputs['pred_semantic_left_up'], outputs['pred_semantic_right_up'], outputs['pred_semantic_right_down'], outputs['pred_semantic_left_down']
        semantic_left_up_probability = out_semantic_left_up.sigmoid()
        semantic_right_up_probability = out_semantic_right_up.sigmoid()
        semantic_right_down_probability = out_semantic_right_down.sigmoid()
        semantic_left_down_probability = out_semantic_left_down.sigmoid()


        scores, topk_indices = torch.topk(edge_probability.view(1, -1), 100, dim=1)
        topk_points = topk_indices // out_edges.shape[2]
        edges = topk_indices % out_edges.shape[2]


        last_edges = last_edge_probability[:, topk_points.squeeze(0), :]
        last_edges = torch.argmax(last_edges, dim=2)

        this_edges = this_edge_probability[:, topk_points.squeeze(0), :]
        this_edges = torch.argmax(this_edges, dim=2)


        semantic_left_up = semantic_left_up_probability[:, topk_points.squeeze(0), :]
        semantic_left_up = torch.argmax(semantic_left_up, dim=2)
        semantic_right_up = semantic_right_up_probability[:, topk_points.squeeze(0), :]
        semantic_right_up = torch.argmax(semantic_right_up, dim=2)
        semantic_right_down = semantic_right_down_probability[:, topk_points.squeeze(0), :]
        semantic_right_down = torch.argmax(semantic_right_down, dim=2)
        semantic_left_down = semantic_left_down_probability[:, topk_points.squeeze(0), :]
        semantic_left_down = torch.argmax(semantic_left_down, dim=2)


        points = torch.gather(points, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        points = points * scale_fct[:, None, :]

        results = [{'scores': s, 'points': b, 'edges': e, 'last_edges': l_e, 'this_edges': t_e,
                    'semantic_left_up': slu, 'semantic_right_up': sru, 'semantic_right_down': srd, 'semantic_left_down': sld}
                   for s, b, e, l_e, t_e, slu, sru, srd, sld in
                   zip(scores, points, edges, last_edges, this_edges, semantic_left_up, semantic_right_up, semantic_right_down, semantic_left_down)]

        return results