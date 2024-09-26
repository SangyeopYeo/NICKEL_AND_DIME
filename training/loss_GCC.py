# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, sync, gain
    ):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

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
        
def gram(x):

        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (c * h * w)
        return G

class StyleGAN2Loss(Loss):
    def __init__(self, device, img_resolution, G_mapping, G_synthesis, D, G_teacher_mapping, G_teacher_synthesis, D_teacher, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
kd_content_lambda=1.0, kd_gram_lambda=100.0, kd_l1_lambda=0.0, kd_res=None, online_distillation=True):
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
        self.pl_mean_t = torch.zeros([], device=device)

        # Generator-Discriminator Cooperative Compression hyperparameters
        self.G_teacher_mapping = G_teacher_mapping
        self.G_teacher_synthesis = G_teacher_synthesis
        self.D_teacher = D_teacher

        self.kd_content_lambda = kd_content_lambda
        self.kd_gram_lambda = kd_gram_lambda
        self.kd_l1_lambda = kd_l1_lambda
        self.kd_res_generator = kd_res['generator']
        self.kd_res_discriminator = kd_res['discriminator']
        self.online_distillation = online_distillation        
                
        self.D_arch_diff = torch.zeros([], device=device)
        self.D_arch_diff_new = []

    def run_G(self, z, c, sync, return_intermediates=None):
        intermediates, intermediates_teacher = None, None
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing"):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]),)
                    z_temp = torch.randn_like(z)
                    ws[:, cutoff:] = self.G_mapping(z_temp, c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            if return_intermediates is not None:
                img, intermediates = self.G_synthesis(ws, return_intermediates)
            else:
                img = self.G_synthesis(ws)

        with misc.ddp_sync(self.G_teacher_mapping, sync):
            z = z.clone().detach()
            c = c.clone().detach()
            z_temp = z_temp.clone().detach()
            ws_teacher = self.G_teacher_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function("style_mixing_teacher"):
                    ws_teacher[:, cutoff:] = self.G_teacher_mapping(z_temp, c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_teacher_synthesis, sync):
            if return_intermediates is not None:
                img_teacher, intermediates_teacher = self.G_teacher_synthesis(ws_teacher, return_intermediates)
            else:
                img_teacher = self.G_teacher_synthesis(ws_teacher)
            
        return img, ws, img_teacher, ws_teacher, intermediates, intermediates_teacher

    def run_D(self, img, img_t, c, sync, return_intermediates=None):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
            img_t = self.augment_pipe(img_t)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
            
        
        logits_dummy = 0
        intermediates_t = 0 
        intermediates_s = 0 
        with misc.ddp_sync(self.D_teacher, sync):
            if return_intermediates is not None:
                logits_t, intermediates_t = self.D_teacher(img_t, c, return_intermediates)
                logits_dummy, intermediates_s = self.D_teacher(img, c, return_intermediates)
            else:
                logits_t = self.D_teacher(img_t, c)
                
        return logits, logits_t, intermediates_s, intermediates_t, logits_dummy

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_teacher_img, _gen_teacher_ws, gen_intermediates, gen_intermediates_teacher  = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), return_intermediates=self.kd_res_generator) # May get synced by Gpl.
                gen_logits, gen_logits_t, dis_intermediates, dis_intermediates_teacher, logits_dummy = self.run_D(gen_img, gen_teacher_img, gen_c, sync=False, return_intermediates=self.kd_res_discriminator)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + 0 * logits_dummy.sum() # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                if self.online_distillation:
                    training_stats.report('Loss/scores_teacher/fake_t', gen_logits_t)
                    training_stats.report('Loss/signs_teacher/fake_t', gen_logits_t.sign())
                    loss_Gmain_t = torch.nn.functional.softplus(-gen_logits_t) # -log(sigmoid(gen_logits))
                    training_stats.report('Loss/G_teacher/loss', loss_Gmain_t)
                    
            with torch.autograd.profiler.record_function('Generator-Discriminator Cooperative Compression_kd_loss'):
                loss_kd_gram = 0.0
                loss_kd_content = 0.0
                for res in self.kd_res_generator:
                    loss_kd_content += torch.sqrt(F.mse_loss(gen_intermediates[res].to(torch.float32), gen_intermediates_teacher[res].detach().to(torch.float32)))
                    loss_kd_gram += torch.sqrt(F.mse_loss(gram(gen_intermediates[res].to(torch.float32)), gram(gen_intermediates_teacher[res].detach().to(torch.float32))))
                
                for res in self.kd_res_discriminator:
                    loss_kd_content += torch.sqrt(F.mse_loss(dis_intermediates[res].to(torch.float32), dis_intermediates_teacher[res].detach().to(torch.float32)))
                    loss_kd_gram += torch.sqrt(F.mse_loss(gram(dis_intermediates[res].to(torch.float32)), gram(dis_intermediates_teacher[res].detach().to(torch.float32))))
                    
                training_stats.report('Loss/G/kd_content_loss', loss_kd_content)
                training_stats.report('Loss/G/kd_gram_loss', loss_kd_gram)
                                        
                loss_Gmain += self.kd_content_lambda * loss_kd_content
                loss_Gmain += self.kd_gram_lambda * loss_kd_gram
                # 
                if self.kd_l1_lambda > 0.0:
                    loss_kd_l1 = F.l1_loss(gen_img, gen_teacher_img.detach())
                    training_stats.report('Loss/G/kd_l1_loss', loss_kd_l1)
                    loss_Gmain += self.kd_l1_lambda * loss_kd_l1
                    
            # G main loss backward    
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                if self.online_distillation:
                    loss_Gmain_t.mean().mul(gain).backward()
                    
        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                # gen_img, gen_ws, gen_teacher_img, gen_teacher_ws, gen_intermediates, gen_intermediates_teacher = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync, return_intermediates=self.kd_res_generator)
                gen_img, gen_ws, gen_teacher_img, gen_teacher_ws, _, _ = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                # loss_Gpl = pl_penalty * self.pl_weight + 0 * sum([tmp.sum() for tmp in gen_intermediates.values()])
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
                
                if self.online_distillation:
                    pl_noise_t = torch.randn_like(gen_teacher_img) / np.sqrt(gen_teacher_img.shape[2] * gen_teacher_img.shape[3])
                    with torch.autograd.profiler.record_function('pl_grads_teacher'), conv2d_gradfix.no_weight_gradients():
                        pl_grads_t = torch.autograd.grad(outputs=[(gen_teacher_img * pl_noise_t).sum()], inputs=[gen_teacher_ws], create_graph=True, only_inputs=True)[0]                        
                    pl_lengths_t = pl_grads_t.square().sum(2).mean(1).sqrt()
                    pl_mean_t = self.pl_mean_t.lerp(pl_lengths_t.mean(), self.pl_decay)
                    self.pl_mean_t.copy_(pl_mean_t.detach())
                    pl_penalty_t = (pl_lengths_t - pl_mean_t).square()
                    
                    training_stats.report('Loss/pl_penalty_teacher', pl_penalty)
                    # loss_Gpl_t = pl_penalty_t * self.pl_weight + 0 * sum([tmp.sum() for tmp in gen_intermediates_teacher.values()])
                    loss_Gpl_t = pl_penalty_t * self.pl_weight
                    training_stats.report('Loss/G_teacher/reg', loss_Gpl_t)
                    
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
                if self.online_distillation:
                    (gen_teacher_img[:, 0, 0, 0] * 0 + loss_Gpl_t).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_teacher_img, _gen_teacher_ws, _, _ = self.run_G(gen_z, gen_c, sync=False)
                gen_logits, gen_logits_t, _, _, _ = self.run_D(gen_img, gen_teacher_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/scores_teacher/fake', gen_logits_t)
                training_stats.report('Loss/signs_teacher/fake', gen_logits_t.sign())
                loss_Dgen_t = torch.nn.functional.softplus(gen_logits_t) # -log(1 - sigmoid(gen_logits))
                    
                loss_gen = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_gen_t = torch.nn.functional.softplus(-gen_logits_t) # -log(sigmoid(gen_logits))
                                
                loss_local = F.l1_loss(loss_gen, loss_Dgen, reduction='none')
                loss_local_target = F.l1_loss(loss_gen_t, loss_Dgen_t).clone().detach()
                self.D_arch_diff_new.append(loss_local_target)
                
                loss_global = (loss_local-self.D_arch_diff).abs().mean()
                    
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward(retain_graph=True)
                if self.online_distillation:
                    loss_Dgen_t.mean().mul(gain).backward()
                    
                self.D.module.set_netD_grad(False)
                loss_global.mean().mul(gain).backward()
                self.D.module.set_netD_grad(True)
                training_stats.report("Loss/D/loss_global", loss_global)
                
                
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_img_tmp_t = real_img_tmp.clone()
                real_logits, real_logits_t, _, _, _ = self.run_D(real_img_tmp, real_img_tmp_t, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/scores_teacher/real', real_logits_t)
                training_stats.report('Loss/signs_teacher/real', real_logits_t.sign())

                loss_Dreal = 0
                loss_Dreal_t = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)                    
                    if self.online_distillation:
                        loss_Dreal_t = torch.nn.functional.softplus(-real_logits_t) # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D_teacher/loss', loss_Dgen_t + loss_Dreal_t)
                                
                loss_Dr1 = 0
                loss_Dr1_t = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
                    
                    if self.online_distillation:
                        with torch.autograd.profiler.record_function('r1_grads_teacher'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits_t.sum()], inputs=[real_img_tmp_t], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1_t = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty_teacher', r1_penalty)
                        training_stats.report('Loss/D_teacher/reg', loss_Dr1_t)
                        
            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                if self.online_distillation:
                    (real_logits_t * 0 + loss_Dreal_t + loss_Dr1_t).mean().mul(gain).backward()
                    
                
    def get_D_arch_diff(self):
        D_arch_diff_new = sum(self.D_arch_diff_new) / len(self.D_arch_diff_new)
        self.D_arch_diff_new = []
        return D_arch_diff_new
    
    def D_arch_diff_update(self, D_arch_diff):
        self.D_arch_diff = D_arch_diff
        

#----------------------------------------------------------------------------
