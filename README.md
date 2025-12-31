## 安装
```bash
python -m pip install -e rl_sim_env-amp_vae_vit/source/rl_sim_env
```

## 训练

```bash
conda activate isaaclab

# 可选：清理残留
fuser -k -9 /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3
```

### CMDP / 约束强化学习（FPPO / PPO-Lagrange / CPO / PCPO / FOCPO）

#### 单卡训练

```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
    --task Rl-Sim-Env-AmpVae-Grq20-V2d3-v0 \
    --run_name grq20_v2d3_fppo \
    --headless \
    --num_envs 2000
```

#### 多卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
    --task Rl-Sim-Env-AmpVae-Grq20-V2d3-v0 \
    --run_name grq20_v2d3_fppo_dist \
    --distributed \
    --headless \
    --num_envs 2500
```

说明：
- 使用 `torchrun` 启动，`--nproc_per_node` 与可用 GPU 数一致；多节点时再加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。
- `--distributed` 开启多卡分布式，脚本会根据 `local_rank` 自动设置 `env_cfg.sim.device` 与 `agent_cfg.device`。
- `num_envs` 表示每张卡上创建的环境数；多卡总环境数约等于 `num_envs * nproc_per_node`。根据显存和负载自行调整。
- 约束算法通过任务的 `rsl_rl_cfg_entry_point` 配置选择，确保 `algorithm.name` 设为 `fppo`（或 `ppo_lagrange` / `cpo` / `pcpo` / `focpo`），并设置 `cost_limit`、`cost_gamma`、`cost_lam`、`desired_kl`、`step_size`、`delta_safe` 等参数。
- 配置入口位于任务目录的 `config_summary.py`，例如 `rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_sim_env/tasks/manager_based/amp_vae/config/grq20_v2d3/config_summary.py`。
- 环境需在 `infos["cost"]` 中提供逐步成本信号（多约束请先求和），训练日志会记录 `Train/cost_limit_margin`。

#### 单卡可视化

```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/play.py \
    --task Rl-Sim-Env-AmpVae-Grq20-V2d3-Play-v0 \
    --num_envs 25 \
    --checkpoint logs/rsl_rl/grq20_v2d3_fppo/
```

#### 多卡可视化（可选）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    rl_sim_env-amp_vae_vit/scripts/rsl_rl/play.py \
    --task Rl-Sim-Env-AmpVae-Grq20-V2d3-Play-v0 \
    --num_envs 25 \
    --checkpoint logs/rsl_rl/grq20_v2d3_fppo/
```

## 代码改动概览（CMDP 重构）

- 算法侧移除 AMP/VAE/RND/Symmetry 训练逻辑，重构为通用 CMDP 训练框架（纯 FPPO）。
- `ActorCritic` 新增独立 `Cost Critic` Head，完全解耦 Reward / Cost 特征。
- `RolloutStorage` 扩展为 `cost_rewards / cost_values / cost_returns / cost_advantages`，并支持 Cost-GAE。
- `OnPolicyRunner` 从 `infos["cost"]` 读取逐步成本，记录 `Train/cost_limit_margin` 等日志指标。
- 新增算法：`FPPO`（投影 + 步长回溯）、`PPO-Lagrange`、`CPO`、`PCPO`、`FOCPO`，支持 `algorithm.name` 统一切换。
- 约束项统一写入 `infos["log"]`，并聚合为 `infos["cost"]` 供算法端直接消费。
- 配置字段更新：新增 `cost_limit / cost_gamma / cost_lam / max_grad_norm / step_size / delta_safe / focpo_eta / focpo_lambda` 等参数。

## 如何新增算法（后续扩展）

1. 在 `rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_algorithms/rsl_rl/algorithms/` 新建算法文件并实现类。
2. 在 `rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_algorithms/rsl_rl/algorithms/__init__.py` 注册到 `ALGORITHM_REGISTRY`。
3. 若需要新超参，在 `rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_algorithms/rsl_rl_wrapper/rl_cfg.py` 中添加字段，并在任务 `config_summary.py` 中配置。
4. 在本 README 的“代码改动概览”中补充算法名与用途。

## 改进方向：
1. 增加 派生动作奖励，让机器人学会落足点定位在足端半径范围内方差比较小的地方。
2.  设计更合理的成本函数与 `cost_limit` 调度策略（例如分阶段收紧约束）。
3.  支持多约束（多 cost 通道）并探索不同聚合策略（求和/最大值）。

## idea
1. 能不能通过约束让机器人理解约束，最后达到约束几乎不被违反。
