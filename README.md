# Nickel and Diming Your GAN: A Dual-Method Approach to Enhancing GAN Efficiency via Knowledge Distillation

(ECCV 2024) Nickel and Diming Your GAN: A Dual-Method Approach to Enhancing GAN Efficiency via Knowledge Distillation - Official Implementation

Sangyeop Yeo, Yoojin Jang, Jaejun Yoo

[[arXiv]](https://arxiv.org/pdf/2405.11614)[[Project page]](https://sangyeopyeo.github.io/Nickel_and_Diming_Your_GAN/)

Abstract: In this paper, we address the challenge of compressing generative adversarial networks (GANs) for deployment in resource-constrained environments by proposing two novel methodologies: Distribution Matching for Efficient compression (DiME) and Network Interactive Compression via Knowledge Exchange and Learning (NICKEL). DiME employs foundation models as embedding kernels for efficient distribution matching, leveraging maximum mean discrepancy to facilitate effective knowledge distillation. Simultaneously, NICKEL employs an interactive compression method that enhances the communication between the student generator and discriminator, achieving a balanced and stable compression process. Our comprehensive evaluation on the StyleGAN2 architecture with the FFHQ dataset shows the effectiveness of our approach, with NICKEL & DiME achieving FID scores of 10.45 and 15.93 at compression rates of 95.73% and 98.92%, respectively. Remarkably, our methods sustain generative quality even at an extreme compression rate of 99.69%, surpassing the previous state-of-the-art performance by a large margin. These findings not only demonstrate our methodologies' capacity to significantly lower GANs' computational demands but also pave the way for deploying high-quality GAN models in settings with limited resources. Our code will be released soon.


# Installation
```
git clone https://github.com/SangyeopYeo/NICKEL_AND_DIME.git
cd NICKEL_AND_DIME
conda env create -n nickel_and_dime --file environment.yaml
conda activate nickel_and_dime
```

# Requisite
Our method requires both a full-size model (teacher) and a pruned model (student).

We provide checkpoints on our repository, but if you prefer, you can use your own model.

Additionally, you can find models trained on various datasets at https://github.com/NVlabs/stylegan2-ada-pytorch.

## Training
We provide two full-size models trained on the FFHQ dataset and the LSUN-Cat dataset.

Additionally, you can train your own model from scratch as shown below:
```
python train.py --config configs/train
```

## Pruning
You can prune the full-size model. We use the pruning laogirthm provided by content-aware GAN compression by default.
```
python prune.py --config configs/prune/CAGC-Pruning-ratio-0.9-ffhq-256.yml
```

To prune using the GAN slimming algorithm, set "pruning_criterion" to GS.



# Knowledge Distillation
You can configure the settings through the configs to use various algorithms.
## Nickel and Dime
```
python train_DD.py --config configs/finetune/DD-Pruned-ratio-0.9-fhq-256.yaml
```

## Baselines
### GAN Slimming: All-in-One GAN Compression by A Unified Optimization Framework (GS)

### Content-Aware GAN Compression (CAGC)

### Information-Theoretic GAN Compression with Variational Energy-based Model (ITGC)

### Mind the Gap in Distilling StyleGANs (StyleKD)




# Checkpoints
We will provide checkpoints soon.


# Cite this work
If you found this repository useful, please consider giving a star and citation:
```
@inproceedings{yeo2025nickel,
  title={Nickel and Diming Your GAN: A Dual-Method Approach to Enhancing GAN Efficiency via Knowledge Distillation},
  author={Yeo, Sangyeop and Jang, Yoojin and Yoo, Jaejun},
  booktitle={European Conference on Computer Vision},
  pages={104--121},
  year={2025},
  organization={Springer}
}
```
# Reference
Our code is based on the repositories below, and we express our sincere gratitude.

StyleGAN2-ADA Official Code: https://github.com/NVlabs/stylegan2-ada-pytorch

GAN Slimming: https://github.com/VITA-Group/GAN-Slimming

Content Aware GAN Compression: https://github.com/lychenyoko/content-aware-gan-compression

StyleKD: https://github.com/xuguodong03/StyleKD

CLIP: https://github.com/openai/CLIP

DINO: https://github.com/facebookresearch/dino