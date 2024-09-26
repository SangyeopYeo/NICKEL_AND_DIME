import os
import argparse
import sys
import yaml
import time
from configs import parser as _parser
import dnnlib

from metrics import metric_main

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def print_time():
    print("\n\n--------------------------------------")
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("--------------------------------------\n\n")

#----------------------------------------------------------------------------

def get_config_args():
    
    parser = argparse.ArgumentParser(description="GAN Compression")
    
    parser.add_argument(
            '-n', '--dry_run', 
            help='Print training options and exit', action="store_true")
    parser.add_argument(
            "--config",
            required=True,
            help="Config file to use"
        )
    args = parser.parse_args()
        
    override_args = _parser.argv_to_vars(sys.argv)
    yaml_txt = open(args.config).read()
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)
    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)
    
    return  args

#----------------------------------------------------------------------------

def setup_training_loop_kwargs():
    # # General options.
    # @click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
    # @click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
    # @click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
    # @click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
    # @click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
    # @click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
    # 
    # # Dataset.
    # @click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
    # @click.option('--cond', help='Train conditional model based on dataset labels [default: false]', type=bool, metavar='BOOL')
    # @click.option('--subset', help='Train with only N images [default: all]', type=int, metavar='INT')
    # @click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')
    # 
    # # Base config.
    # @click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
    # @click.option('--gamma', help='Override R1 gamma', type=float)
    # @click.option('--kimg', help='Override training duration', type=int, metavar='INT')
    # @click.option('--batch', help='Override batch size', type=int, metavar='INT')
    # 
    # # Discriminator augmentation.
    # @click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
    # @click.option('--p', help='Augmentation probability for --aug=fixed', type=float)
    # @click.option('--target', help='ADA target value for --aug=ada', type=float)
    # @click.option('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))
    # 
    # # Transfer learning.
    # @click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
    # @click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')
    # 
    # # Performance options.
    # @click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
    # @click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
    # @click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
    # @click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
    # @click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')
    
    args = get_config_args()
    kwargs = ['outdir', 'num_gpus', 'snap', 'metrics', 'random_seed',
              'data', 'cond', 'subset', 'mirror',
              'cfg', 'gamma', 'kimg', 'batch',
              'aug', 'p', 'target', 'augpipe',
              'resume', 'freezed',
              'fp32', 'nhwc', 'allow_tf32', 'nobench', 'workers',
              'teacher_ckpt', 'student_ckpt', 'sparse_loss', 'sparsity_eta', 'kd_percept_mode', 'kd_percept_lambda', 'kd_l1_mode', 'kd_l1_lambda', 'pruning_ratio'
              ]

    for kwarg in kwargs:
        if kwarg not in args:
            setattr(args, kwarg, None)

    # ------------------------------------------
    # General options: num_gpus, snap, metrics, random_seed
    # ------------------------------------------

    assert type(args.outdir) is str

    if args.num_gpus is None:
        args.num_gpus = 1
    assert isinstance(args.num_gpus, int)
    if not (args.num_gpus >= 1 and args.num_gpus & (args.num_gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')

    if args.snap is None:
        args.snap = 50
    assert isinstance(args.snap, int)
    if args.snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = args.snap
    args.network_snapshot_ticks = args.snap

    if args.metrics is None:
        args.metrics = ['fid50k_full']
    assert isinstance(args.metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    if args.random_seed is None:
        args.random_seed = 0
    assert isinstance(args.random_seed, int)

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert args.data is not None
    assert isinstance(args.data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=args.data, use_labels=True, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = os.path.splitext(os.path.basename(args.config))[0]
        desc += f'-{training_set.name}'
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    if args.cond is None:
        args.cond = False
    assert isinstance(args.cond, bool)
    if args.cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if args.subset is not None:
        assert isinstance(args.subset, int)
        if not 1 <= args.subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{args.subset}'
        if args.subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = args.subset
            args.training_set_kwargs.random_seed = args.random_seed

    if args.mirror is None:
        args.mirror = False
    assert isinstance(args.mirror, bool)
    if args.mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if args.cfg is None:
        args.cfg = 'auto'
    assert isinstance(args.cfg, str)
    desc += f'-{args.cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert args.cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[args.cfg])
    if args.cfg == 'auto':
        desc += f'{args.num_gpus:d}'
        spec.ref_gpus = args.num_gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(args.num_gpus * min(4096 // res, 32), 64), args.num_gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // args.num_gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if args.cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if args.gamma is not None:
        assert isinstance(args.gamma, float)
        if not args.gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{args.gamma:g}'
        args.loss_kwargs.r1_gamma = args.gamma

    if args.kimg is not None:
        assert isinstance(args.kimg, int)
        if not args.kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{args.kimg:d}'
        args.total_kimg = args.kimg

    if args.batch is not None:
        assert isinstance(args.batch, int)
        if not (args.batch >= 1 and args.batch % args.num_gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{args.batch}'
        args.batch_size = args.batch
        args.batch_gpu = args.batch // args.num_gpus


    # ---------------------------------------------------
    # GAN Slimming settings: teacker_ckpt, student_ckpt, sparse_loss, sparsity_eta, kd_percept_mode, kd_percept_lambda, kd_l1_mode, kd_l1_lambda, pruning_ratio
    # ---------------------------------------------------

    assert args.teacher_ckpt is None or isinstance(args.teacher_ckpt, str)
    assert args.student_ckpt is None or isinstance(args.student_ckpt, str)

    if args.sparse_loss is None:
        args.sparse_loss = False
    assert isinstance(args.sparse_loss, bool)    
    
    if args.sparsity_eta is None:
        args.sparsity_eta = 0.0
    args.sparsity_eta = float(args.sparsity_eta)
    assert isinstance(args.sparsity_eta, float)
    
    assert args.kd_percept_mode in [None, 'VGG', 'LPIPS']
    
    if args.kd_percept_lambda is None:
        args.kd_percept_lambda = 0.0
    args.kd_percept_lambda = float(args.kd_percept_lambda)
    assert isinstance(args.kd_percept_lambda, float)
    
    assert args.kd_l1_mode in [None, 'Output_Only', 'Intermediate']
    
    if args.kd_l1_lambda is None:
        args.kd_l1_lambda = 0.0
    args.kd_l1_lambda = float(args.kd_l1_lambda)
    assert isinstance(args.kd_l1_lambda, float)
    
    if args.pruning_ratio is None:
        args.pruning_ratio = 0.0
    assert isinstance(args.pruning_ratio, float) and 0.0 <= args.pruning_ratio <= 1.0

    # Check whether training is for GAN Slimming Sparsity or not
    assert args.sparse_loss ^ (args.pruning_ratio!=0.0)
    
    # Set GAN Slimming Loss kwargs
    args.loss_kwargs.class_name = 'training.loss_GS.StyleGAN2Loss'
    args.loss_kwargs.sparse_loss = args.sparse_loss
    args.loss_kwargs.sparsity_eta = args.sparsity_eta
    args.loss_kwargs.kd_percept_mode = args.kd_percept_mode
    args.loss_kwargs.kd_percept_lambda = args.kd_percept_lambda
    args.loss_kwargs.kd_l1_mode = args.kd_l1_mode
    args.loss_kwargs.kd_l1_lambda = args.kd_l1_lambda
    
    # When finetune pruned network
    args.G_kwargs.class_name = 'training.networks_GS.Generator'
    args.G_teacher_kwargs = args.G_kwargs.copy()
    args.G_teacher_kwargs['pruning_ratio'] = 0.0
    args.G_student_kwargs = args.G_kwargs.copy()
    args.G_student_kwargs['pruning_ratio'] = args.pruning_ratio
    del args.G_kwargs

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if args.aug is None:
        args.aug = 'ada'
    else:
        assert isinstance(args.aug, str)
        desc += f'-{args.aug}'

    if args.aug == 'ada':
        args.ada_target = 0.6

    elif args.aug == 'noaug':
        pass

    elif args.aug == 'fixed':
        if args.p is None:
            raise UserError(f'--aug={args.aug} requires specifying --p')

    else:
        raise UserError(f'--aug={args.aug} not supported')

    if args.p is not None:
        assert isinstance(args.p, float)
        if args.aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= args.p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{args.p:g}'
        args.augment_p = args.p

    if args.target is not None:
        assert isinstance(args.target, float)
        if args.aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= args.target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{args.target:g}'
        args.ada_target = args.target

    assert args.augpipe is None or isinstance(args.augpipe, str)
    if args.augpipe is None:
        args.augpipe = 'bgc'
    else:
        if args.aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{args.augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert args.augpipe in augpipe_specs
    if args.aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[args.augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert args.resume is None or isinstance(args.resume, str)
    if args.resume is None:
        args.resume = 'noresume'
    elif args.resume == 'noresume':
        desc += '-noresume'
    elif args.resume in resume_specs:
        desc += f'-resume{args.resume}'
        args.resume_pkl = resume_specs[args.resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = args.resume # custom path or url

    if args.resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if args.freezed is not None:
        assert isinstance(args.freezed, int)
        if not args.freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{args.freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = args.freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if args.fp32 is None:
        args.fp32 = False
    assert isinstance(args.fp32, bool)
    if args.fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if args.nhwc is None:
        args.nhwc = False
    assert isinstance(args.nhwc, bool)
    if args.nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if args.nobench is None:
        args.nobench = False
    assert isinstance(args.nobench, bool)
    if args.nobench:
        args.cudnn_benchmark = False

    if args.allow_tf32 is None:
        args.allow_tf32 = False
    assert isinstance(args.allow_tf32, bool)

    if args.workers is not None:
        assert isinstance(args.workers, int)
        if not args.workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = args.workers

    return desc, args

#----------------------------------------------------------------------------