import os
import numpy as np
import torch
import random

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = False
    # this causes the code to vary across runs. I don't want that for now.
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))