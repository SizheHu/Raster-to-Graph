import argparse

import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float) # 2e-4->-5
    parser.add_argument('--lr_backbone_names', default=["backbone_with_position_embedding.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float) # 2e-5->-6
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int) # 16
    parser.add_argument('--weight_decay', default=1e-5, type=float)  # 1e-5
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')  # 0.1
    parser.add_argument('--optim', default='Adam', help='SGD Adam')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_maps', default=4, type=int, help='number of feature levels')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes for object detection')
    parser.add_argument('--num_classes_edges', default=14, type=int)

    # Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float)

    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=20, type=int)
    parser.add_argument('--enc_n_points', default=20, type=int)

    # Hungarian Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost, 2")
    parser.add_argument('--set_cost_point', default=20, type=float,
                        help="L1 point coefficient in the matching cost, 20")
    parser.add_argument('--set_cost_edge', default=2, type=float,
                        help="Edge coefficient in the matching cost, 2")
    parser.add_argument('--set_cost_last_edge', default=2, type=float,
                        help="Edge coefficient in the matching cost, 2")
    parser.add_argument('--set_cost_this_edge', default=2, type=float,
                        help="Edge coefficient in the matching cost, 2")
    parser.add_argument('--set_cost_semantic', default=0.5, type=float,
                        help="Edge coefficient in the matching cost, 2")



    parser.add_argument('--cls_loss_coef', default=2, type=float, help='')
    parser.add_argument('--point_loss_coef', default=20, type=float, help='')
    parser.add_argument('--edge_loss_coef', default=2, type=float, help='')
    parser.add_argument('--last_edge_loss_coef', default=2, type=float, help='')
    parser.add_argument('--this_edge_loss_coef', default=2, type=float, help='')
    parser.add_argument('--semantic_loss_coef', default=0.5, type=float, help='')

    parser.add_argument('--random_region', default=True, type=bool)


    parser.add_argument('--dataset_path', default='./data/dataset_v5', type=str)

    parser.add_argument('--output_dir', default='./output/testingaa',
                        help='例：./output/output_r2v_gtlayer2_multiscale'
                             'empty for no saving')
    # cpu/cuda
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=42) # 42

    parser.add_argument('--resume', default='./output/trained_r2gmodel/checkpoint0299.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser
