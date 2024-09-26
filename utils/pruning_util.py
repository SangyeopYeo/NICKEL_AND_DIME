import numpy as np
from utils.content_aware_pruning import get_content_aware_pruning_scores


def get_pruning_scores(generator, discriminator, args, device):
    if args.pruning_criterion == "CAGC":
        score_list = get_content_aware_pruning_scores(generator = generator,
                                                      args = args,
                                                      device = device)
    
    return score_list

def get_uniform_rmvelist(net_shape, pruning_ratio):
    rmve_list = (np.array(net_shape) * pruning_ratio).astype(int)
    return rmve_list

def get_default_mask_from_shape(net_shape):
    net_default_mask = [np.array([True] * layer_shape) for layer_shape in net_shape]
    return net_default_mask


def generate_prune_mask_list(net_score_list, net_shape, rmve_list):
    net_mask_list = get_default_mask_from_shape(net_shape)
    print('\n' + '-----------------------------Actual Pruning Happens-----------------------------')
    
    for lay_k in range(len(net_shape)):
        layer_mask = net_mask_list[lay_k]
        layer_rmv = rmve_list[lay_k]
        layer_score_list = net_score_list[lay_k]
        assert len(layer_mask) == len(layer_score_list)

        print("\n" + "Layer ID: " + str(lay_k))
        print("Layer Remove: " + str(layer_rmv))
        
        if (sum(layer_mask) > layer_rmv and layer_rmv > 0):
            rmv_node = np.argsort(layer_score_list)[:layer_rmv]
            layer_mask[rmv_node] = False

            print('We have masked out  #' + str(rmv_node) + ' in layer ' + str(lay_k) + '. It will have ' + str(sum(layer_mask)) + " maps.")

    return net_mask_list
