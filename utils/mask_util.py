from copy import deepcopy
from .network_util import get_generator_kernel_key, get_network_shape
from .pruning_util import get_uniform_rmvelist, generate_prune_mask_list



def mask_the_generator(model_dict, pruning_score, args):

    net_shape = get_network_shape(model_dict)
    rmve_list = get_uniform_rmvelist(net_shape, args.pruning_ratio)
    
    print(net_shape)
    print('#'*25)
    print(rmve_list)
    print('#'*25)
    
    if args.pruning_criterion == 'GS':
        _pruning_score = []
        idx = 0
        for shape in net_shape:
            _pruning_score.append(pruning_score[idx:idx+shape])
            idx+=shape
        pruning_score = _pruning_score
    
    prune_net_mask = generate_prune_mask_list(pruning_score, net_shape, rmve_list)
    
    generator_key = get_generator_kernel_key(model_dict)
    
    pruned_dict = deepcopy(model_dict)
    
    pruned_dict["synthesis.b4.const"] = model_dict["synthesis.b4.const"].cpu()[prune_net_mask[0], ...]
    
    mask_generator_key(model_dict, pruned_dict, prune_net_mask, **generator_key)       

    return pruned_dict


def mask_generator_key(model_dict, pruned_dict, net_mask_list, conv_key, torgb_key):
    mask_conv_key(model_dict, pruned_dict, net_mask_list, conv_key)
    mask_torgb_key(model_dict, pruned_dict, net_mask_list, torgb_key)


def mask_conv_key(model_dict, pruned_dict, net_mask_list, conv_key):

    for idx in range(len(conv_key) // 4):
        
        input_mask, output_mask = net_mask_list[idx], net_mask_list[idx + 1]
        weight_key, bias_key, affine_weight_key, affine_bias_key = conv_key[idx * 4:idx * 4 + 4]
        pruned_dict[weight_key]        = model_dict[weight_key].cpu()[output_mask, ...][:, input_mask, ...]
        pruned_dict[bias_key]          = model_dict[bias_key].cpu()[output_mask]
        pruned_dict[affine_weight_key] = model_dict[affine_weight_key].cpu()[input_mask, ...]
        pruned_dict[affine_bias_key]   = model_dict[affine_bias_key].cpu()[input_mask]

def mask_torgb_key(model_dict, pruned_dict, net_mask_list, torgb_key):

    for idx in range(len(torgb_key) // 4):
        
        layer_mask = net_mask_list[2*idx + 1]
        weight_key, bias_key, affine_weight_key, affine_bias_key = torgb_key[idx*4:(idx+1)*4]
        pruned_dict[weight_key]        = model_dict[weight_key].cpu()[:, layer_mask, ...]
        pruned_dict[bias_key]          = model_dict[bias_key].cpu()
        pruned_dict[affine_weight_key] = model_dict[affine_weight_key].cpu()[layer_mask, ...]
        pruned_dict[affine_bias_key]   = model_dict[affine_bias_key].cpu()[layer_mask]
