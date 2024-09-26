﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
from torch.nn import functional as F
import dnnlib
import torch.distributed as dist
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [
                indices[(i + gw) % len(indices)] for i in range(len(indices))
            ]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
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
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_teacher_kwargs={},  # Options for teacher generator network.
    G_student_kwargs={},  # Options for student generator network.
    D_teacher_kwargs={},  # Options for teacher discriminator network.
    D_student_kwargs={},  # Options for student discriminator network.optimizer.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    online_distillation=True, # Options for teacher / student online distillaion.
    kd_res=None, # Options for teacher / student distillation resolutions.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=None,  # EMA ramp-up coefficient.
    G_reg_interval=4,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
    resume_pkl=None,  # Network pickle to resume training from.
    resume_train_pkl=None,  # Network pickle to continue training from.
    teacher_ckpt=None,  # Teacher Generator pickle to load from.
    student_ckpt=None,  # Student Generator pickle to load from.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    allow_tf32=False,  # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.,
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = (
        allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    )
    torch.backends.cudnn.allow_tf32 = (
        allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    )
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print("Loading training set...")
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs
    )  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed
    )
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=batch_size // num_gpus,
            **data_loader_kwargs,
        )
    )
    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution,  img_channels=training_set.num_channels,)
    G = dnnlib.util.construct_class_by_name(kd_res=kd_res, **G_student_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_student_kwargs, **common_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_teacher = dnnlib.util.construct_class_by_name(**G_teacher_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    D_teacher = dnnlib.util.construct_class_by_name(**D_teacher_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G.initialize()
    G.train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()    
    if online_distillation:
        G_teacher.train()
        D_teacher.train()
        G_teacher_ema = copy.deepcopy(G_teacher).eval()
    
    D_masked = dnnlib.util.construct_class_by_name(class_name="training.networks_GCC.MaskDiscriminator", discriminator=D, threshold=0.1).train().requires_grad_(False).to(device)
    D_masked.set_threshold_device(device)
    
    # Loading teacher / student model weights
    if (teacher_ckpt is not None) and (rank == 0):
        print(f'Loading teacher generator from "{teacher_ckpt}"')
        with dnnlib.util.open_url(teacher_ckpt) as f:
            resume_data = legacy.load_network_pkl(f)
        misc.copy_params_and_buffers(resume_data["G_ema"], G_teacher, require_all=False)
        misc.copy_params_and_buffers(resume_data["D"], D_teacher, require_all=False)

    if (student_ckpt is not None) and (rank == 0):
        print(f'Loading student generator / discriminator from "{teacher_ckpt}"')
        with dnnlib.util.open_url(student_ckpt) as f:
            resume_data = legacy.load_network_pkl(f)
        # When pruning generator, teacher discriminator is saved together
        for name, module in [("G", G), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [("G", G), ("D", D_masked), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            if online_distillation:
                for name, module in [("G_teacher", G_teacher), ("D_teacher", D_teacher), ("G_teacher_ema", G_teacher_ema)]:
                    misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Continue from existing pickle.
    if resume_train_pkl is not None:
        print(f'Continuing from "{resume_train_pkl}"')
        with dnnlib.util.open_url(resume_train_pkl) as f:
            resume_data = legacy.load_last_pkl(f)

    # # Print network summary tables.
    # if rank == 0:
    #     z = torch.empty([batch_gpu, G.z_dim], device=device)
    #     c = torch.empty([batch_gpu, G.c_dim], device=device)
    #     img = misc.print_module_summary(G, [z, c])
    #     _ = misc.print_module_summary(G_teacher, [z, c])
    #     misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Distribute across GPUs.
    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    ddp_modules = dict()

    net_list = [("G_mapping", G.mapping), ("G_synthesis", G.synthesis), ("D", D_masked), (None, G_ema), ("augment_pipe", augment_pipe),] # ddp_modules for StyleGAN2 / StyleGAN2-ADA
    net_list += [("G_teacher_mapping", G_teacher.mapping), ("G_teacher_synthesis", G_teacher.synthesis), ("D_teacher", D_teacher), (None, G_teacher_ema)]  # ddp modules for distilling

    for name, module in net_list:
        if ((num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0):
            module.requires_grad_(True)
            if name == "G_synthesis":
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
            else:
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module
        
    # Setup training phases.
    if rank == 0:
        print("Setting up training phases...")
    loss = dnnlib.util.construct_class_by_name(
        device=device,
        img_resolution=training_set.resolution,
        **ddp_modules,
        **loss_kwargs,
    )  # subclass of training.loss.Loss
    phases = []
        
    for name, module, opt_kwargs, reg_interval in [("G", (G, G_teacher), G_opt_kwargs, G_reg_interval), ("D", (D_masked, D_teacher), D_opt_kwargs, D_reg_interval), ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module[0].parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer

            # Continue train from existing pickle.
            if resume_train_pkl is not None:
                opt.load_state_dict(resume_data[f"{name}_opt"])

            _dict = dnnlib.EasyDict(name=name + "both", module=module[0], opt=opt, interval=1)
            
            if online_distillation:
                _dict.module_teacher = module[1]
                opt_teacher = dnnlib.util.construct_class_by_name(params=module[1].parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer

                # Continue train from existing pickle.
                if resume_train_pkl is not None:
                    opt_teacher.load_state_dict(resume_data[f"{name}_teacher_opt"])
                
                _dict.opt_teacher = opt_teacher
                
            phases += [_dict]
            
            if name == 'D':
                D_mask_opt_kwargs = opt_kwargs
            
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module[0].parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer

            if name == 'D':
                D_mask_opt_kwargs = opt_kwargs
                opt = dnnlib.util.construct_class_by_name(module[0].discriminator.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer

            # Continue train from existing pickle.
            if resume_train_pkl is not None:
                opt.load_state_dict(resume_data[f"{name}_opt"])

            _dict = dnnlib.EasyDict(name=name + "main", module=module[0], opt=opt, interval=1)
            _dict_reg = dnnlib.EasyDict(name=name + "reg", module=module[0], opt=opt, interval=reg_interval)

            if online_distillation:
                _dict.module_teacher = module[1]
                _dict_reg.module_teacher = module[1]
                
                opt_teacher = dnnlib.util.construct_class_by_name(module[1].parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
                
                # Continue train from existing pickle.
                if resume_train_pkl is not None:
                    opt_teacher.load_state_dict(resume_data[f"{name}_teacher_opt"])

                _dict.opt_teacher = opt_teacher
                _dict_reg.opt_teacher = opt_teacher

            phases += [_dict]
            phases += [_dict_reg]
            
                
    D_mask_opt_kwargs.lr = D_mask_opt_kwargs.lr / 8.
    opt_D_masked = dnnlib.util.construct_class_by_name(D_masked.masks.parameters(), **D_mask_opt_kwargs)  # subclass of torch.optim.Optimizer
    if resume_train_pkl is not None:
        opt_D_masked.load_state_dict(resume_data[f"D_mask_opt"])
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        if resume_train_pkl is None:
            save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size,)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        if resume_train_pkl is None:
            save_image_grid(images, os.path.join(run_dir, "fakes_init.png"), drange=[-1, 1], grid_size=grid_size,)

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
        # try:
        #     import torch.utils.tensorboard as tensorboard

        #     stats_tfevents = tensorboard.SummaryWriter(run_dir)
        # except ImportError as err:
        #     print("Skipping tfevents export:", err)

    # Train.
    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    # Continue train from existing pickle.
    if resume_train_pkl is not None:
        cur_nimg = resume_data["cur_nimg"]
        tick_start_nimg = cur_nimg
        cur_tick = resume_data["cur_tick"]
        batch_idx = resume_data["batch_idx"]
        for phase in phases:
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    
    D_arch_diff = torch.zeros([], device=device)
    gather_t = [torch.ones_like(D_arch_diff) for _ in range(dist.get_world_size())]
    
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function("data_fetch"):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
                
            phase.opt.zero_grad(set_to_none=True)
            if 'D' in phase.name:
                opt_D_masked.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            
            if online_distillation:
                phase.opt_teacher.zero_grad(set_to_none=True)
                phase.module_teacher.requires_grad_(True)
            
            # if rank==0:
            #     print(phase.name)
            #     print('G_student: ', G.synthesis.b128.conv1.weight.sum(), '||', G.synthesis.b128.conv1.weight.grad.sum() if G.synthesis.b128.conv1.weight.grad is not None else None)
            #     print('G_student transfer: ', G.synthesis.transfer128.weight.sum(), '||', G.synthesis.transfer128.weight.grad.sum() if G.synthesis.transfer128.weight.grad is not None else None)
            #     print('D_student: ', D.b128.conv1.weight.sum(), '||', D.b128.conv1.weight.grad.sum() if D.b128.conv1.weight.grad is not None else None)
            #     print('D_student mask: ', D_masked.m16.alpha.sum(), '||', D_masked.m16.alpha.grad.sum() if D_masked.m16.alpha.grad is not None else None)
            #     print('G_teacher: ', G_teacher.synthesis.b128.conv1.weight.sum(), '||', G_teacher.synthesis.b128.conv1.weight.grad.sum() if G_teacher.synthesis.b128.conv1.weight.grad is not None else None)
            #     print('D_teacher: ', D_teacher.b128.conv1.weight.sum(), '||', D_teacher.b128.conv1.weight.grad.sum() if D_teacher.b128.conv1.weight.grad is not None else None)
            #     print('-'*25)          
            
            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = round_idx == batch_size // (batch_gpu * num_gpus) - 1
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain,)
                
            # Update weights.
            phase.module.requires_grad_(False)  
            
            # if rank==0:
            #     print(phase.name)
            #     print('G_student: ', G.synthesis.b128.conv1.weight.sum(), '||', G.synthesis.b128.conv1.weight.grad.sum() if G.synthesis.b128.conv1.weight.grad is not None else None)
            #     print('G_student transfer: ', G.synthesis.transfer128.weight.sum(), '||', G.synthesis.transfer128.weight.grad.sum() if G.synthesis.transfer128.weight.grad is not None else None)
            #     print('D_student: ', D.b128.conv1.weight.sum(), '||', D.b128.conv1.weight.grad.sum() if D.b128.conv1.weight.grad is not None else None)
            #     print('D_student mask: ', D_masked.m16.alpha.sum(), '||', D_masked.m16.alpha.grad.sum() if D_masked.m16.alpha.grad is not None else None)
            #     print('G_teacher: ', G_teacher.synthesis.b128.conv1.weight.sum(), '||', G_teacher.synthesis.b128.conv1.weight.grad.sum() if G_teacher.synthesis.b128.conv1.weight.grad is not None else None)
            #     print('D_teacher: ', D_teacher.b128.conv1.weight.sum(), '||', D_teacher.b128.conv1.weight.grad.sum() if D_teacher.b128.conv1.weight.grad is not None else None)
            #     print('='*25)
            
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            if 'D' in phase.name:
                opt_D_masked.step()
                D_masked.clipping_mask_alpha()
            phase.opt.step()

            if online_distillation:
                phase.module_teacher.requires_grad_(False)         
                with torch.autograd.profiler.record_function(phase.name + "_teacher_opt"):
                    for param in phase.module_teacher.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    phase.opt_teacher.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function("Gema"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
                
            if online_distillation:
                with torch.autograd.profiler.record_function("G_teacher_ema"):
                    for p_ema, p in zip(G_teacher_ema.parameters(), G_teacher.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    for b_ema, b in zip(G_teacher_ema.buffers(), G_teacher.buffers()):
                        b_ema.copy_(b)

        D_arch_diff_new = loss.get_D_arch_diff()
        dist.all_gather(gather_t, D_arch_diff_new)
        D_arch_diff_new = sum(gather_t) / len(gather_t)
        D_arch_diff.copy_(D_arch_diff_new.lerp(D_arch_diff, ema_beta))
        __temp = D_arch_diff.clone()
        torch.distributed.broadcast(tensor=__temp, src=0)
        assert (D_arch_diff == __temp).all()    
        loss.D_arch_diff_update(D_arch_diff)
        training_stats.report('Loss/D/D_arch_diff', D_arch_diff)
        
        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = (np.sign(ada_stats["Loss/signs/real"] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000))
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = cur_nimg >= total_kimg * 1000
        if ((not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)):
            continue
        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
        training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(" ".join(fields))
        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")
        # Save image snapshot.
        if ((rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0)):
            images = torch.cat([G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"), drange=[-1, 1], grid_size=grid_size,)
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            
            net_list = [("G", G), ("D", D_masked), ("G_ema", G_ema), ("augment_pipe", augment_pipe),]
            if online_distillation:
                net_list+=[("G_teacher", G_teacher), ("D_teacher", D_teacher), ("G_teacher_ema", G_teacher_ema)]
            
            for name, module in net_list:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r".*\.w_avg")
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl")
            if rank == 0:
                with open(snapshot_pkl, "wb") as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print("Evaluating metrics...")
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data["G_ema"], dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device,)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0("Timing/" + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + "\n")
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f"Metrics/{name}", value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        ######### Added for continuing train #########
        snapshot_last_pkl = None
        snapshot_last_data = None
        if (network_snapshot_ticks is not None) and (done or (cur_tick - 1) % network_snapshot_ticks == 0):
            snapshot_last_data = dict(cur_nimg=cur_nimg, cur_tick=cur_tick, batch_idx=batch_idx, snapshot_pkl=os.path.basename(snapshot_pkl),)
            for phase in phases:
                if "main" in phase.name:
                    snapshot_last_data[f"{phase.name[0]}_opt"] = phase.opt.state_dict()
                    if online_distillation:
                        snapshot_last_data[f"{phase.name[0]}_teacher_opt"] = phase.opt_teacher.state_dict()
            
            snapshot_last_data[f"D_mask_opt"] = opt_D_masked.state_dict()
            snapshot_last_pkl = os.path.join(run_dir, f"network-snapshot-last.pkl")
            if rank == 0:
                with open(snapshot_last_pkl, "wb") as f:
                    pickle.dump(snapshot_last_data, f)
        ######### Added for continuing train #########

        if done:
            break

    # Done.
    if rank == 0:
        print()
        print("Exiting...")


# ----------------------------------------------------------------------------
