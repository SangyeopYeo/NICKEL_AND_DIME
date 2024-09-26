import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path
from utils.face_parsing.BiSeNet import BiSeNet

from .network_util import convert_tensor_to_image
from PIL import Image

file_path = Path(__file__).parent

def get_parsing_net(device):
    
    PRETRAINED_FILE = (file_path / '''./face_parsing/pretrained_model/79999_iter.pth''').resolve()

    n_classes = 19
    parsing_net = BiSeNet(n_classes = n_classes).to(device)
    pretrained_weight = torch.load(PRETRAINED_FILE, map_location=device)
    parsing_net.load_state_dict(pretrained_weight)
    parsing_net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    return parsing_net, to_tensor
    

import PIL
def save_image_grid(img, fname, drange, grid_size):
    img = img.cpu().detach().numpy()
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        
def get_content_aware_pruning_scores(generator, args, device):
    generator.requires_grad_(True)
    parsing_net, to_tensor = get_parsing_net(device)

    n_batch = args.n_sample // args.batch_size
    batch_size_list = [args.batch_size] * (n_batch - 1) + [args.batch_size + args.n_sample % args.batch_size]
    grad_score_list = []

    for (idx, batch) in enumerate(batch_size_list):
        print("Processing Batch: " + str(idx))
        z = torch.randn(batch, generator.z_dim, device = device)
        c = torch.empty(batch, generator.c_dim, device = device)
        img_tensor = generator(z, c)  
        img_size = img_tensor.shape[-1]
        
        noisy_img_list = []
        for i in range(batch):
            single_img = img_tensor[i:i+1,...]

            pil_single_img = convert_tensor_to_image(single_img)
            parsing = extract_face_mask(pil_single_img, parsing_net, to_tensor, device)
            mask = (parsing > 0) * (parsing != 16)
            resized_mask = np.array(Image.fromarray(mask).resize((img_size, img_size)))

            noisy_img = get_salt_pepper_noisy_image(single_img, resized_mask, args.noise_prob)
            noisy_img_list.append(noisy_img)
        
        noisy_img_tensor = torch.cat(noisy_img_list)        
        grad_score = get_weight_gradient(noisy_img_tensor, img_tensor, generator)
        grad_score_list.append(grad_score)
        generator.zero_grad()

    return grad_score_list

def extract_face_mask(pil_image, parsing_net, to_tensor, device):
    with torch.no_grad():
        image = pil_image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = parsing_net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    return parsing    

def get_salt_pepper_noisy_image(img_tensor, mask, noise_prob):
    img_size = img_tensor.shape[-1]
    salt_pepper_noise = np.random.randint(low = 0, high = 2, size = (img_size, img_size)) * 2 - 1

    noisy_img = img_tensor.clone()
    for h in range(img_size):
        for w in range(img_size):
            if mask[h,w] == True and (np.random.random() < noise_prob):
                noisy_img[:, :, h, w] = salt_pepper_noise[h,w]

    return noisy_img

def get_weight_gradient(noisy_img, img_tensor, model):
    loss = torch.sum(torch.abs(noisy_img - img_tensor))
    loss.backward()

    module_list = []

    for n, m in model.synthesis.named_modules():
        if hasattr(m, "conv0"):
            module_list.append(m.conv0)
        if hasattr(m, "conv1"):
            module_list.append(m.conv1)
    module_list.append(model.synthesis.b256.torgb)
    grad_list = [module.weight.grad for module in module_list]
    
    grad_score_list = [(torch.mean(torch.abs(grad), axis = [0,2,3])).detach().cpu().numpy() for grad in grad_list]
    
    return grad_score_list