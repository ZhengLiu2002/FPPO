# RL SIM ENV

## Submodule
```bash
git submodule update --init --recursive
```

## Conda Environment Setup
```bash
conda create -n [env_name] python=3.10
conda activate [env_name]
conda install onnxruntime
pip install --upgrade pip
pip install ruamel.yaml
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# verifying the isaac sim
isaacsim

# in other path !!!!
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 93fd2120cf95fa2e4b37bca855b63fe965d3d344
./isaaclab.sh --install

# in rl_sim_env path !!!!
pip install --upgrade wandb
python -m pip install -e source/rl_sim_env
```

## Wandb Setup

Sign up to wandb: https://wandb.ai/ and get the API key.

Run the following command to login to wandb:
```bash

wandb login
export WANDB_USERNAME=USER_NAME
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Show Available Environments
```bash
python scripts/list_envs.py
```

## Train

### single-gpu
```bash
python scripts/[env_name]/train.py --task [task_name] --run_name [run_name] --headless
```
### multi-gpu
```bash
export JAX_LOCAL_RANK=[start_gpu_id]
python -m torch.distributed.run --nnodes=[server_num] --nproc_per_node=[gpu_num] scripts/[env_name]/train.py --task [task_name] --run_name [run_name] --device cuda:[start_gpu_id] --headless --distributed
```
