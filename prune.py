from main_utils_DD import setup_training_loop_kwargs, print_time, UserError
from utils.pruning_util import get_pruning_scores
from utils.mask_util import mask_the_generator
from utils.utils import set_random_seed

import pickle
import copy
import time
import os
import torch
import shutil
import json
import numpy as np
import legacy
import dnnlib
from torch_utils import misc


def main():
    _, args = setup_training_loop_kwargs()     
    # Print options.
    print()
    print('Beginning of Pruning process')
    print(json.dumps(vars(args), indent=4))
    print()
    print(f'Training data:      {args.training_set_kwargs.path}')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print()   
    print_time()
    set_random_seed(args.random_seed)
    device = torch.device('cuda')
    ####
    
    print('Constructing networks...')
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = os.path.splitext(os.path.basename(args.config))[0]
        desc += f'-{training_set.name}'
    except IOError as err:
        raise UserError(f'--data: {err}')
    
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)

    G_ema = dnnlib.util.construct_class_by_name(**args.G_teacher_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**args.D_teacher_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    print(f'full model loading from "{args.load_ckpt}"')
    with dnnlib.util.open_url(args.load_ckpt) as f:
        load_data = legacy.load_network_pkl(f)
    for name, module in [('D', D), ('G_ema', G_ema)]:
        if module is not None:
            misc.copy_params_and_buffers(load_data[name], module, require_all=False) # dense model load
        
    start_time = time.time()
    score_list = get_pruning_scores(generator = G_ema,
                                    discriminator = D, 
                                    args = args,
                                    device = device)
    score_array = np.array([np.array(score) for score in score_list])    
    pruning_score = np.sum(score_array, axis=0)    
    end_time = time.time()
    
    print("The %s criterion scoring takes: " %args.pruning_criterion, str(round(end_time - start_time, 4)) + ' seconds')

    pruned_generator_dict = mask_the_generator(G_ema.state_dict(), pruning_score, args)
    
    G_pruned = dnnlib.util.construct_class_by_name(**args.G_student_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_pruned.load_state_dict(pruned_generator_dict)

    for name, module in [('G', G_pruned), ('D', D), ('G_ema', G_pruned)]:
        if module is not None:
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        load_data[name] = module
        del module # conserve memory
    
    # Save path settings    
    save_path = os.path.join(args.outdir, f'pruning-ratio-{args.pruning_ratio}')
    
    if args.cond:
        desc += '-cond'
    if args.subset is not None:
        desc += f'-subset{args.subset}'
    if args.mirror:
        desc += '-mirror'
    desc += f'-{args.cfg}'
    if args.kimg is not None:
        assert isinstance(args.kimg, int)
        if not args.kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{args.kimg:d}'
    desc += f'-{args.aug}'
    if args.p is not None:
        desc += f'-p{args.p:g}'
    if args.target is not None:
        desc += f'-target{args.target:g}'
    if not args.aug=='noaug' :
        desc += f'-{args.augpipe}'
    
    save_path = os.path.join(save_path, desc)    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'pruning_options.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)    
    snapshot_pkl = os.path.join(save_path, f'pruned_network-{args.pruning_ratio}.pkl')

    with open(snapshot_pkl, 'wb') as f:
        pickle.dump(load_data, f)

    print()
    print('Exiting...')

if __name__== "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if os.path.exists("/root/.cache/torch_extensions"):
            shutil.rmtree("/root/.cache/torch_extensions")