import torch
from torchvision import utils
from PIL import Image


def get_generator_kernel_key(model_dict):

    conv_key_list = get_conv_kernel_key(model_dict)
    torgb_key_list = get_torgb_kernel_key(model_dict)

    generator_key = {"conv_key": conv_key_list, "torgb_key": torgb_key_list}
    return generator_key


def get_conv_kernel_key(model_dict):
    conv_key_list = []

    for key in model_dict.keys():
        if ("conv" in key) and (("weight" in key) or ("bias" in key)):
            conv_key_list.append(key)
    
    return conv_key_list


def get_torgb_kernel_key(model_dict):
    torgb_key_list = []

    for key in model_dict.keys():
        if "torgb" in key:
            torgb_key_list.append(key)

    return torgb_key_list


def convert_tensor_to_image(img_tensor):

    grid = utils.make_grid(img_tensor, nrow = 1, padding = 2, pad_value = 0,
                            normalize = True, range = (-1, 1), scale_each = False)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def get_network_shape(model_dict):
    conv_key_list = get_conv_kernel_key(model_dict)
    num_channels = [model_dict[key].shape[1] for key in conv_key_list if (("affine" not in key) and ("bias" not in key))]
    num_channels.append(model_dict[conv_key_list[-4]].shape[0])
    return num_channels