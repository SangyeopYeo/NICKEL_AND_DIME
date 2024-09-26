python prune.py --config configs/prune/CAGC-Pruning-ratio-0.9-ffhq-256.yml
python train_DD.py --config configs/finetune/DD-Pruned-ratio-0.9-fhq-256.yaml

conda env create --file environment.yaml