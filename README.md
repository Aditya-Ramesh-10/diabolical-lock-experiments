# Diabolical Lock experiments from RC-GVF

This repository supplements the paper ["Exploring through Random Curiosity with General Value Functions"](https://arxiv.org/abs/2211.10282).

The implementation focuses on clarity and flexibility rather than computational efficiency.


## Instructions

Run an individual experiment with RC-GVF from the root directory:
```bash
python3 -m scripts.db_lock_train_rcgvf

# Or with RND
python3 -m scripts.db_lock_train_rnd
```

The settings for environment and algorithm can be modified in the `config_defaults` dictionary or through a [Weights & Biases (wandb)](https://docs.wandb.ai/) sweep. In case you don't want to utilise Weights & Biases:

```bash
export WANDB_MODE=disabled
```


## Dependencies

- gymnasium==0.26.3
- torch-ac
- numpy
- torch
- scipy
- wandb

## Acknowledgements

The code follows the structure used in [rl_starter_files](https://github.com/lcswillems/rl-starter-files). The diabolical lock environment is based on the description in https://arxiv.org/abs/1911.05815.