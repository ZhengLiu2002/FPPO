# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import numpy as np
import torch


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_amp_vae_policy_as_onnx(
    actor: object,
    vae: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxAmpVaeExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


def export_amp_vae_perception_policy_as_onnx(
    actor: object,
    vae: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxAmpVaeExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class OnnxAmpVaeExporter(torch.nn.Module):
    """
    Exporter of combined VAE encoder and Actor network to a single ONNX file,
    integrating VAE output into Actor input, with dynamic batch support.
    """

    def __init__(self, actor, vae, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor = copy.deepcopy(actor).cpu()
        self.vae = copy.deepcopy(vae).cpu()

    def forward(self, actor_obs, vae_obs):
        # 1) VAE encoding (deterministic)
        code = self.vae.act_inference(vae_obs)
        # 2) Concatenate VAE code and actor observations
        full_obs = torch.cat((code, actor_obs), dim=-1)
        # 3) Actor network produces actions
        actions = self.actor.act_inference(full_obs)
        return actions

    def export(self, path, filename):
        """Export the combined model to ONNX at path/filename"""
        self.cpu()
        full_path = os.path.join(path, filename)

        # 1) Infer full input dimension for actor (code + actor_obs)
        actor_in = None
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in = module.in_features
                break
        if actor_in is None:
            raise RuntimeError("Unable to infer actor input dimension")

        # 2) Infer VAE observation dimension from first Linear in encoder
        vae_in = None
        for module in self.vae.encoder.modules():
            if isinstance(module, torch.nn.Linear):
                vae_in = module.in_features
                break
        if vae_in is None:
            raise RuntimeError("Unable to infer VAE input dimension")

        # 3) Compute code dimension by running dummy through VAE
        dummy_vae = torch.zeros(1, vae_in)
        with torch.no_grad():
            code = self.vae.act_inference(dummy_vae)
        code_dim = code.shape[1]

        # 4) Actor obs dimension = actor_in - code_dim
        actor_obs_dim = actor_in - code_dim
        if actor_obs_dim <= 0:
            raise RuntimeError(f"Inferred actor_obs_dim={actor_obs_dim} invalid")
        dummy_actor = torch.zeros(1, actor_obs_dim)

        # Specify dynamic axes for batch dimension
        dynamic_axes = {
            "actor_obs": {0: "batch_size"},
            "vae_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_actor, dummy_vae),
            full_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["actor_obs", "vae_obs"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved ONNX combined Actor+VAE model to {full_path}")


def export_inference_cfg(env, env_cfg, path, load_run, checkpoint):
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]
    policy_cfg_dict["input_names"] = ["actor_obs", "vae_obs"]
    policy_cfg_dict["output_names"] = ["actions"]
    policy_cfg_dict["input_actor_obs_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    policy_cfg_dict["input_vae_obs_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    input_actor_obs_scales = {}
    input_vae_obs_scales = {}
    input_obs_size_map = {}
    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation
    input_actor_obs_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_actor_obs_scales["projected_gravity"] = 1.0
    input_actor_obs_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_actor_obs_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_actor_obs_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_actor_obs_scales["actions"] = 1.0

    input_vae_obs_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_vae_obs_scales["projected_gravity"] = 1.0
    input_vae_obs_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_vae_obs_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_vae_obs_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_vae_obs_scales["actions"] = 1.0

    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    input_obs_size_map["vae_obs"] = env_cfg.num_vae_obs

    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["input_vae_obs_scales"] = input_vae_obs_scales
    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs
    actor_obs_history_length = env.unwrapped.observation_manager._group_obs_term_cfgs["actor_obs"][1].history_length
    vae_obs_history_length = env.unwrapped.cfg.config_summary.env.obs_history_length
    policy_cfg_dict["obs_history_length"] = {
        "actor_obs": actor_obs_history_length if actor_obs_history_length > 0 else 1,
        "vae_obs": vae_obs_history_length if vae_obs_history_length > 0 else 1,
    }
    kp = env.unwrapped.scene.articulations["robot"].actuators["base_legs"].stiffness[0].cpu().numpy()
    kd = env.unwrapped.scene.articulations["robot"].actuators["base_legs"].damping[0].cpu().numpy()
    policy_cfg_dict["joint_kp"] = [float(f"{x:.4f}") for x in kp.flatten()]
    policy_cfg_dict["joint_kd"] = [float(f"{x:.4f}") for x in kd.flatten()]
    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    print("input_vae_obs_names:", policy_cfg_dict["input_vae_obs_names"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_vae_obs_scales:", policy_cfg_dict["input_vae_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    export_inference_cfg_to_yaml(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml(config_dict, path, load_run, checkpoint):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in (
        ("actor_obs", config_dict["input_actor_obs_names"]),
        ("vae_obs", config_dict["input_vae_obs_names"]),
    ):
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in (
        ("actor_obs", config_dict["input_actor_obs_scales"]),
        ("vae_obs", config_dict["input_vae_obs_scales"]),
    ):
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"
    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)
