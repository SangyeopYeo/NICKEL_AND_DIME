﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torchvision import transforms
from torch.nn import functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from utils.perceptual import VGGFeature, perceptual_loss
from utils.content_aware_pruning import get_parsing_net
import lpips

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

import PIL
def save_image_grid(img, fname, drange, grid_size):
    img = img.cpu().detach()
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

def Langevin_dynamics(ebm, teacher_image, student_image, sigma_t, langevin_steps, step_size):
        fake_teacher_img_MCMC = torch.autograd.Variable(student_image.data, requires_grad=True)
        for k in range(langevin_steps):
            E_prime = torch.autograd.grad(ebm(fake_teacher_img_MCMC, student_image).sum(),
                                        [fake_teacher_img_MCMC], retain_graph=True)[0]
            fake_teacher_img_MCMC.data = fake_teacher_img_MCMC.data - step_size * E_prime \
                + sigma_t * torch.randn_like(fake_teacher_img_MCMC)

        fake_teacher_img_MCMC = fake_teacher_img_MCMC.detach()
        return fake_teacher_img_MCMC

def batch_img_parsing(img_tensor, parsing_net, device, transform=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), PARSING_SIZE=512):    
    scale_factor = PARSING_SIZE / img_tensor.shape[-1]
    transformed_tensor = ((img_tensor + 1 ) / 2).clamp(0,1) # Rescale tensor to [0,1]
    transformed_tensor = F.interpolate(transformed_tensor, 
                                       scale_factor=scale_factor, 
                                       mode='bilinear', 
                                       align_corners=False,
                                       recompute_scale_factor=True) # Scale to 512
    transformed_tensor = transform(transformed_tensor).to(device)
    with torch.no_grad():
        img_parsing = parsing_net(transformed_tensor)[0]
    parsing = img_parsing.argmax(1)
    return parsing

def get_masked_tensor(img_tensor, batch_parsing, device, mask_grad=False, PARSING_SIZE=512):
    '''
    Usage:
        To produce the masked img_tensor in a differentiable way
    
    Args:
        img_tensor:    (torch.Tensor) generated 4D tensor of shape [N,C,H,W]
        batch_parsing: (torch.Tensor) the parsing result from SeNet of shape [N,512,512] (the net fixed the parsing to be 512)
        device:        (str) the device to place the tensor
        mask_grad:     (bool) whether requires gradient to flow to the masked tensor or not
    '''
    mask = (batch_parsing > 0) * (batch_parsing != 16)
    mask_float = mask.type(torch.FloatTensor)[:, None, :, :] # Make it to a 4D tensor with float for interpolation    
    scale_factor = img_tensor.shape[-1] / PARSING_SIZE
    
    resized_mask = F.interpolate(mask_float, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
    resized_mask = (resized_mask > 0.5).type(torch.FloatTensor).to(device)
    
    if mask_grad:
        resized_mask.requires_grad = True
    masked_img_tensor = torch.zeros_like(img_tensor).to(device)
    masked_img_tensor = img_tensor * resized_mask
    
    return masked_img_tensor

class StyleGAN2Loss(Loss):
    def __init__(self, device, img_resolution, G_mapping, G_synthesis, D, G_teacher_mapping, G_teacher_synthesis, ebm=None, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, kd_percept_mode='LPIPS', kd_percept_lambda=3.0, kd_l1_mode='Output_Only', kd_l1_lambda=0.0, kd_VID_langevin_steps=10, kd_VID_step_size=50.0, kd_VID_sigma_t=0.005, kd_VID_MI_lambda=0.05, kd_VID_l2_lambda=0.05):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        
        # Content Aware GAN Compression with VEM hyperparameters
        self.G_teacher_mapping = G_teacher_mapping
        self.G_teacher_synthesis = G_teacher_synthesis
        self.ebm = ebm
        self.kd_percept_mode = kd_percept_mode
        self.kd_percept_lambda = kd_percept_lambda
        self.kd_l1_mode = kd_l1_mode
        self.return_rgb_list = kd_l1_mode == 'Intermediate'
        self.kd_l1_lambda = kd_l1_lambda        
        self.kd_img_resolution = 256
        self.kd_img_downsample = img_resolution > self.kd_img_resolution # pooled the image for LPIPS for memory saving
        self.pooled_kernel_size = img_resolution // self.kd_img_resolution
        self.parsing_net, _ = get_parsing_net(device)
        self.kd_VID_langevin_steps = kd_VID_langevin_steps
        self.kd_VID_step_size = kd_VID_step_size
        self.kd_VID_sigma_t = kd_VID_sigma_t
        self.kd_VID_MI_lambda = kd_VID_MI_lambda
        self.kd_VID_l2_lambda = kd_VID_l2_lambda
        
        if self.kd_percept_mode == 'VGG':
            self.percept_loss = VGGFeature().eval().to(device)
        if self.kd_percept_mode == 'LPIPS':
            self.percept_loss = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[device])
        

    def run_G(self, z, c, sync, do_main=False):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    z_temp = torch.randn_like(z)
                    ws[:, cutoff:] = self.G_mapping(z_temp, c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            styles_list = []
            student_rgb_img_list = []
            if self.return_rgb_list:
                img, _, student_rgb_img_list = self.G_synthesis(ws, False, self.return_rgb_list)
            else:
                img = self.G_synthesis(ws)
                
        if not do_main:
            return img, ws
        
        with misc.ddp_sync(self.G_teacher_mapping, sync):
            ws_teacher = self.G_teacher_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing_teacher'):
                    ws_teacher[:, cutoff:] = self.G_teacher_mapping(z_temp, c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_teacher_synthesis, sync):
            teacher_rgb_img_list = []
            if self.return_rgb_list:
                img_teacher, _, teacher_rgb_img_list = self.G_teacher_synthesis(ws_teacher, False, self.return_rgb_list)
            else:
                img_teacher = self.G_teacher_synthesis(ws_teacher)
        return img, ws, img_teacher, ws_teacher, styles_list, student_rgb_img_list, teacher_rgb_img_list

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_teacher_img, _gen_teacher_ws, styles_list, student_rgb_img_list, teacher_rgb_img_list = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), do_main=True) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                    
            with torch.autograd.profiler.record_function('Content_Aware_GAN_Compression_kd_loss'):
                # kd_VID_loss
                if self.ebm is not None:         
                    with misc.ddp_sync(self.ebm, sync):       
                        fake_teacher_img_MCMC = Langevin_dynamics(self.ebm, gen_teacher_img, gen_img, self.kd_VID_sigma_t, self.kd_VID_langevin_steps, self.kd_VID_step_size)
                        plus_output = self.ebm(gen_teacher_img, gen_img)
                        minus_output = self.ebm(fake_teacher_img_MCMC, gen_img)
                    kd_MI_loss = plus_output - minus_output
                    training_stats.report('Loss/G/kd_MI_loss', kd_MI_loss)
                    loss_Gmain = loss_Gmain + self.kd_VID_MI_lambda * kd_MI_loss
                    
                # kd_l1_loss
                if self.kd_l1_mode == 'Output_Only':
                    gen_teacher_img_parsing = batch_img_parsing(gen_teacher_img, self.parsing_net, self.device)    
                    gen_teacher_img_mask = get_masked_tensor(gen_teacher_img, gen_teacher_img_parsing, self.device, mask_grad = False)
                    gen_img_mask = get_masked_tensor(gen_img, gen_teacher_img_parsing, self.device, mask_grad = True)
                    kd_l1_loss = torch.mean(torch.abs(gen_teacher_img_mask - gen_img_mask))
                    training_stats.report('Loss/G/kd_l1_loss', kd_l1_loss)
                    loss_Gmain = loss_Gmain + self.kd_l1_lambda * kd_l1_loss
                elif self.kd_l1_mode == 'Intermediate':
                    kd_l1_loss_list = [torch.mean(torch.abs(teacher_rgb_img - student_rgb_img)) for (teacher_rgb_img, student_rgb_img) in zip(teacher_rgb_img_list, student_rgb_img_list)] 
                    kd_l1_loss = sum(kd_l1_loss_list)
                    training_stats.report('Loss/G/kd_l1_loss', kd_l1_loss)
                    loss_Gmain = loss_Gmain + self.kd_l1_lambda * kd_l1_loss
                    
                # kd_percept_loss
                if self.kd_img_downsample:
                    gen_img = F.avg_pool2d(gen_img, kernel_size = self.pooled_kernel_size, stride = self.pooled_kernel_size)
                    gen_teacher_img = F.avg_pool2d(gen_teacher_img, kernel_size = self.pooled_kernel_size, stride = self.pooled_kernel_size)
                if self.kd_percept_mode == 'VGG':
                    student_output_vgg_features = self.percept_loss(gen_img_mask)
                    teacher_output_vgg_features = self.percept_loss(gen_teacher_img_mask)
                    kd_percept_loss = perceptual_loss(student_output_vgg_features, teacher_output_vgg_features)[0]
                    training_stats.report('Loss/G/kd_VGG_loss', kd_percept_loss)
                    loss_Gmain = loss_Gmain + self.kd_percept_lambda * kd_percept_loss
                elif self.kd_percept_mode == 'LPIPS':
                    kd_percept_loss = torch.mean(self.percept_loss(gen_img_mask, gen_teacher_img_mask))
                    training_stats.report('Loss/G/kd_LPIPS_loss', kd_percept_loss)
                    loss_Gmain = loss_Gmain + self.kd_percept_lambda * kd_percept_loss
                    
            # G main loss backward    
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
            
            if self.ebm is not None:
                with torch.autograd.profiler.record_function('ebm_forward'):
                    gen_img, _gen_ws, gen_teacher_img, _gen_teacher_ws, styles_list, student_rgb_img_list, teacher_rgb_img_list = self.run_G(gen_z, gen_c, sync=False, do_main=True)
                    with misc.ddp_sync(self.ebm, sync):  
                        fake_teacher_img_MCMC = Langevin_dynamics(self.ebm, gen_teacher_img, gen_img, self.kd_VID_sigma_t, self.kd_VID_langevin_steps, self.kd_VID_step_size)
                        plus_output = self.ebm(gen_teacher_img, gen_img)
                        minus_output = self.ebm(fake_teacher_img_MCMC, gen_img)
                    loss_VEM = plus_output.mean() - minus_output.mean() + self.kd_VID_l2_lambda * (plus_output ** 2 + minus_output ** 2).mean()
                    training_stats.report('Loss/VEM/loss', loss_VEM)
                with torch.autograd.profiler.record_function('ebm_backward'):
                    loss_VEM.backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
