## FPPO 代码库说明

该仓库用于验证 FPPO 及多种约束强化学习算法在复杂地形机器人任务中的效果。已移除 AMP / VAE / VIT 相关逻辑，仅保留 FPPO 所需的通用组件和 CMDP 训练流程。

### 代码清理与重构要点
- 删除 AMP/VAE/VIT 相关脚本、环境与任务定义，仅保留 FPPO 训练所需模块。
- 新增 `FPPOEnv`，在环境中引入 cost 计算并写入 `infos["cost"]`。
- 任务目录重构为 `tasks/manager_based/fppo`，配置统一使用 `fppo_base_env_cfg.py`。
- 约束项统一写入 cost：默认包含稳定性与扭矩限制。

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

### 教师-学生蒸馏（盲走学生）
- 准备：让环境输出教师观测；简单做法是在 `FPPOEnv` 的观测收集处把 `extras["observations"]["teacher"] = extras["observations"]["critic"]`（或在配置里新增 `teacher` 观测组等于 `critic_obs`）。教师 actor 可临时打开高度扫描等特权项；学生保持默认盲走输入。
- 阶段 A：训练教师（全感知 FPPO）
```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
  --task Isaac-FPPO-Grq20-V2d3-v0 \
  --run_name teacher_full \
  --headless \
  --num_envs 4096 \
  --log_project_name isaaclab-fppo
```
- 阶段 B：蒸馏到盲学生（Distillation + StudentTeacher）
```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
  --task Isaac-FPPO-Grq20-V2d3-v0 \
  --run_name student_distill \
  --headless \
  --num_envs 4096 \
  algorithm.name=distillation \
  policy.class_name=StudentTeacher \
  load_run=teacher_full \
  load_checkpoint=model_.*.pt \
  --log_project_name isaaclab-fppo
```
  说明：Distillation 会把 `load_run` 中的教师权重加载到 teacher 分支，student 分支学习盲走观测→教师动作的映射。若需自定义学生/教师网络宽度，追加 `policy.student_hidden_dims=[512,256,128] policy.teacher_hidden_dims=[512,256,128]`。
- 阶段 C（可选）：用蒸馏学生推理
```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/play.py \
  --task Isaac-FPPO-Grq20-V2d3-Play-v0 \
  --num_envs 25 \
  algorithm.name=distillation \
  policy.class_name=StudentTeacher \
  --checkpoint logs/rsl_rl/student_distill/
```

### Task ID
- `Isaac-FPPO-Grq20-V2d3-v0`

### 单卡训练（示例：V2d3）
```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
    --task Isaac-FPPO-Grq20-V2d3-v0 \
    --run_name grq20_v2d3_fppo \
    --headless \
    --num_envs 4096 \
    --log_project_name isaaclab-fppo
```

### 多卡训练（示例：V2d3）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    rl_sim_env-amp_vae_vit/scripts/rsl_rl/train.py \
    --task Isaac-FPPO-Grq20-V2d3-v0 \
    --run_name grq20_v2d3_fppo_dist \
    --distributed \
    --headless \
    --num_envs 4096 \
    --log_project_name isaaclab-fppo
```

说明：
- 使用 `torchrun` 启动时，`--nproc_per_node` 与可用 GPU 数一致；多节点时再加 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。
- `--distributed` 开启多卡分布式，脚本会根据 `local_rank` 自动设置 `env_cfg.sim.device` 与 `agent_cfg.device`。
- `num_envs` 表示每张卡上创建的环境数；多卡总环境数约等于 `num_envs * nproc_per_node`。

## 可视化
### Task ID
- `Isaac-FPPO-Grq20-V2d3-Play-v0`

### 单卡可视化（示例：V2d3）
```bash
python rl_sim_env-amp_vae_vit/scripts/rsl_rl/play.py \
    --task Isaac-FPPO-Grq20-V2d3-Play-v0 \
    --num_envs 25 \
    --checkpoint logs/rsl_rl/grq20_v2d3_fppo/
```

### 多卡可视化（示例：V2d3）
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    rl_sim_env-amp_vae_vit/scripts/rsl_rl/play.py \
    --task Isaac-FPPO-Grq20-V2d3-Play-v0 \
    --num_envs 25 \
    --checkpoint logs/rsl_rl/grq20_v2d3_fppo/
```


## 诊断脚本

```bash
../isaaclab.sh -p rl_sim_env-amp_vae_vit/scripts/tools/diagnose_fppo_motion.py \
  --task Isaac-FPPO-Grq20-V2d3-v0 \
  --num_envs 64 \
  --steps 200 \
  --action_mode policy \
  --headless \
  --policy_checkpoint logs/rsl_rl/grq20_v2d3_fppo/


../isaaclab.sh -p rl_sim_env-amp_vae_vit/scripts/tools/diagnose_fppo_contacts.py \
  --task Isaac-FPPO-Grq20-V2d3-v0 \
  --num_envs 64 \
  --steps 200 \
  --action_mode policy \
  --headless \
  --policy_checkpoint logs/rsl_rl/grq20_v2d3_fppo/



```

## 配置与算法切换
- 配置入口位于任务目录的 `config_summary.py`：
  - `rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_sim_env/tasks/manager_based/fppo/config/grq20_v2d3/config_summary.py`
- 算法选择：在 `config_summary.py` 中设置 `algorithm.name` 为 `fppo` / `ppo_lagrange` / `cpo` / `pcpo` / `focpo`。
- 成本信号：环境在 `infos["cost"]` 中输出逐步 cost，训练日志记录 `Train/cost_limit_margin` 等指标。

### 成本项（Cost Terms）
- 默认成本项：
  - `stability_constraint`（稳定性 / 倾斜角）
  - `torque_constraint`（关节扭矩限制）
- 成本权重与阈值配置：
  - `ConfigSummary.cost.stability`
  - `ConfigSummary.cost.torque_limit`

## 代码结构
- 环境：`rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_sim_env/envs/fppo_env.py`
- 任务配置：`rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_sim_env/tasks/manager_based/fppo/`
- 约束定义：`rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_sim_env/tasks/manager_based/common/mdp/constraints.py`
- 算法实现：`rl_sim_env-amp_vae_vit/source/rl_sim_env/rl_algorithms/rsl_rl/algorithms/`

## 改进方向
1. 设计更合理的成本函数与 `cost_limit` 调度策略（例如分阶段收紧约束）。
2. 支持多 cost 通道与不同聚合策略（求和/最大值）。
3. 增加更精细的落足点约束或派生动作奖励。
