import random

import numpy as np
import torch

from util import misc


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)