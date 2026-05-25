#!/usr/bin/env python3
"""
Analyze Action-Reasoning Gradient Attribution across LIBERO++ suites (zero-shot).

Uses the LIBERO++ benchmark from libero_plus/. Samples ~100 tasks uniformly
at random across all suites and reports gradient-based modality attribution.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict, deque
import types
import numpy as np
import tqdm

# -- Environment setup (before any libero imports) --
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda", type=int, choices=[0, 1], default=0)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.cuda)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# libero_plus_root = os.path.join(project_root, "libero_plus")
libero_plus_root = '/data/aryaman/DeepThinkVLA_libero_plus'

# Point libero config at LIBERO++ data paths
libero_plus_config_dir = os.path.join(project_root, ".libero_plus")
os.makedirs(libero_plus_config_dir, exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = libero_plus_config_dir

import yaml
libero_plus_libero = os.path.join(libero_plus_root, "libero", "libero")
config_data = {
    "benchmark_root": libero_plus_libero,
    "bddl_files": os.path.join(libero_plus_libero, "bddl_files"),
    "init_states": os.path.join(libero_plus_libero, "init_files"),
    "datasets": os.path.join(libero_plus_libero, "..", "datasets"),
    "assets": os.path.join(libero_plus_libero, "assets"),
}
with open(os.path.join(libero_plus_config_dir, "config.yaml"), "w") as f:
    yaml.dump(config_data, f)

# LIBERO++ benchmark module first, then src/ for model/experiment code
sys.path.insert(0, libero_plus_root)
sys.path.insert(1, os.path.join(project_root, "src"))

import torch
from transformers import AutoProcessor
from libero.libero import benchmark
from experiments.run_libero_plus_eval import (
    GenerateConfig, TASK_MAX_STEPS, get_libero_dummy_action,
    get_libero_env, set_seed_everywhere, prepare_observation
)
from sft.modeling_deepthinkvla import DeepThinkVLA
from experiments.deepthinkvla_utils import _get_unomrmalize_action, get_vla_action
from sft.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

TOTAL_SAMPLES = 100


def evaluate_gradients(checkpoint_path: str, output_path: str):
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))

    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint_path,
        num_images_in_input=2,
        seed=429
    )
    set_seed_everywhere(cfg.seed)

    print(f"Loading DeepThinkVLA from {checkpoint_path} ...")
    vla = DeepThinkVLA.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=getattr(torch, cfg.compute_dtype),
    )
    vla.eval()
    vla = vla.to("cuda:0")
    unomrmalize_action = _get_unomrmalize_action(cfg.pretrained_checkpoint)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)

    # Global tracking variables for monkey patch
    global shared_query_indices, shared_input_ids
    global shared_grad_inputs, shared_embeds, _gradient_mode
    shared_query_indices = None
    shared_input_ids = None
    shared_grad_inputs = None
    shared_embeds = None
    _gradient_mode = False

    old_lm_forward = vla.language_model.forward
    def patched_lm_forward(self, *args, **kwargs):
        global shared_embeds, _gradient_mode
        if _gradient_mode:
            if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
                embeds = kwargs['inputs_embeds'].detach().requires_grad_(True)
                shared_embeds = embeds
                kwargs['inputs_embeds'] = embeds
        return old_lm_forward(*args, **kwargs)
    vla.language_model.forward = types.MethodType(patched_lm_forward, vla.language_model)

    old_prompt_cot_predict = vla.prompt_cot_predict_action
    def patched_prompt_cot_predict(self, input_cot_ids, pixel_values, attention_mask):
        global shared_query_indices, shared_input_ids, shared_grad_inputs
        input_ids_copy = input_cot_ids.clone()
        input_ids_copy, _ = self._prepare_input_for_action_prediction(input_ids_copy, attention_mask.clone())
        sorted_indices = torch.argsort(((input_ids_copy.ne(self.pad_token_id))).int(), dim=1, descending=True, stable=True)
        final_input_ids = torch.gather(input_ids_copy, 1, sorted_indices)
        shared_input_ids = final_input_ids[0].cpu().tolist()

        shared_grad_inputs = (
            input_cot_ids.detach().clone(),
            pixel_values.detach().clone(),
            attention_mask.detach().clone(),
        )

        logits, action_start_idx = old_prompt_cot_predict(input_cot_ids, pixel_values, attention_mask)

        start_indices = action_start_idx.unsqueeze(1)
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)
        seq_indices = start_indices + position_offsets
        shared_query_indices = seq_indices[0].cpu().tolist()

        return logits, action_start_idx
    vla.prompt_cot_predict_action = types.MethodType(patched_prompt_cot_predict, vla)

    def compute_gradient_attribution(action_idxs, visual_idxs, reasoning_idxs, textual_idxs):
        global shared_grad_inputs, shared_embeds, _gradient_mode
        input_cot_ids, pixel_values, attention_mask = shared_grad_inputs

        for p in vla.parameters():
            p.requires_grad_(False)

        _gradient_mode = True
        try:
            logits, _ = vla.prompt_cot_predict_action(
                input_cot_ids, pixel_values, attention_mask
            )

            # Restrict to the 256 physical action bins
            begin_idx = vla.config.action_token_begin_idx
            end_idx = vla.config.action_token_end_idx
            action_logits = logits[0, action_idxs]
            bin_logits = action_logits[:, begin_idx:end_idx+1]
            predicted_ids = bin_logits.argmax(dim=-1)
            scalar = bin_logits[torch.arange(len(predicted_ids)), predicted_ids].mean()

            scalar.backward()

            grad = shared_embeds.grad
            embeds = shared_embeds.detach()
            
            # Compute both L1 and L2 norms
            input_x_grad = embeds * grad
            attribution_l1 = input_x_grad.abs().sum(dim=-1)[0]
            attribution_l2 = torch.norm(input_x_grad, p=2, dim=-1)[0]

            scores = {}
            for name, idxs in [("reasoning", reasoning_idxs), ("visual", visual_idxs),
                               ("textual", textual_idxs), ("action", action_idxs)]:
                scores[f"{name}_l1"] = float(attribution_l1[idxs].mean().item()) if idxs else 0.0
                scores[f"{name}_l2"] = float(attribution_l2[idxs].mean().item()) if idxs else 0.0
            return scores
        finally:
            _gradient_mode = False
            shared_embeds = None
            vla.zero_grad()
            for p in vla.parameters():
                p.requires_grad_(True)

    # ---- Build task pool and sample TOTAL_SAMPLES uniformly ----
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    benchmark_dict = benchmark.get_benchmark_dict()

    all_tasks = []
    suite_task_suites = {}
    for suite_name in suites:
        task_suite = benchmark_dict[suite_name]()
        suite_task_suites[suite_name] = task_suite
        for task_id in range(task_suite.n_tasks):
            all_tasks.append((suite_name, task_id))

    random.seed(cfg.seed)
    sampled_tasks = random.sample(all_tasks, min(TOTAL_SAMPLES, len(all_tasks)))

    print(f"\nSampled {len(sampled_tasks)} tasks from {len(all_tasks)} across {len(suites)} suites")
    suite_dist = defaultdict(int)
    for s, _ in sampled_tasks:
        suite_dist[s] += 1
    for s in suites:
        print(f"  {s}: {suite_dist[s]}")

    # ---- Tracking structures ----
    modalities = ["reasoning", "visual", "textual", "action"]
    all_grad_l1 = {m: [] for m in modalities}
    all_grad_l2 = {m: [] for m in modalities}
    all_successes = []

    # ---- Run evaluation ----
    for idx, (suite_name, task_id) in enumerate(
        tqdm.tqdm(sampled_tasks, desc="LIBERO++ eval")
    ):
        task_suite = suite_task_suites[suite_name]
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        print(f"\n[{idx+1}/{len(sampled_tasks)}] Suite: {suite_name} | Task: {task_description}")
        env.reset()
        obs = env.set_init_state(initial_states[0])
        action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)

        t = 0
        max_steps = TASK_MAX_STEPS[suite_name]
        ep_grad_l1 = {m: [] for m in modalities}
        ep_grad_l2 = {m: [] for m in modalities}
        episode_success = False

        while t < max_steps + 10:
            if t < 10:
                obs, reward, done, info = env.step(get_libero_dummy_action())
                t += 1
                continue

            if len(action_queue) == 0:
                observation, img = prepare_observation(obs, 224)
                with torch.no_grad():
                    actions, cot_text = get_vla_action(
                        cfg=cfg,
                        vla=vla,
                        unomrmalize_action=unomrmalize_action,
                        processor=processor,
                        obs=observation,
                        task_label=task_description
                    )

                action_queue.extend(actions)

                seq_len = len(shared_input_ids)
                image_token = 257152
                think_token = 257153
                end_think_token = 257154

                visual_idxs = [i for i, x in enumerate(shared_input_ids) if x == image_token]

                think_ids = [i for i, x in enumerate(shared_input_ids) if x == think_token]
                end_think_ids = [i for i, x in enumerate(shared_input_ids) if x == end_think_token]
                reasoning_idxs = []
                for start in think_ids:
                    ends = [e for e in end_think_ids if e > start]
                    end = ends[0] if ends else shared_query_indices[0] - 1
                    reasoning_idxs.extend(list(range(start + 1, end)))

                action_idxs = shared_query_indices

                tagged = set(visual_idxs) | set(reasoning_idxs) | set(action_idxs)
                textual_idxs = [i for i in range(seq_len) if i not in tagged]

                if len(action_idxs) > 0:
                    grad_scores = compute_gradient_attribution(
                        action_idxs, visual_idxs, reasoning_idxs, textual_idxs
                    )
                    for m in modalities:
                        ep_grad_l1[m].append(grad_scores[f"{m}_l1"])
                        ep_grad_l2[m].append(grad_scores[f"{m}_l2"])

            action = action_queue.popleft()
            action[..., -1] = np.sign(action[..., -1])
            obs, reward, done, info = env.step(action.tolist())

            if done:
                episode_success = True
                break
            t += 1

        all_successes.append(episode_success)

        if any(ep_grad_l1[m] for m in ep_grad_l1):
            for m in modalities:
                val_grad_l1 = float(np.mean(ep_grad_l1[m])) if ep_grad_l1[m] else 0.0
                val_grad_l2 = float(np.mean(ep_grad_l2[m])) if ep_grad_l2[m] else 0.0
                all_grad_l1[m].append(val_grad_l1)
                all_grad_l2[m].append(val_grad_l2)
            print(f"  -> Grad Reasoning L1: {all_grad_l1['reasoning'][-1]:.6f} | L2: {all_grad_l2['reasoning'][-1]:.6f} | Success: {episode_success}")
        else:
            print(f"  -> Success: {episode_success}")

    # ---- Aggregate results ----
    def _stats(raw):
        return {"mean": float(np.mean(raw)) if raw else 0.0,
                "std":  float(np.std(raw))  if raw else 0.0,
                "raw":  [float(x) for x in raw]}

    metrics = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "gradient_mass_l1":  {m: _stats(all_grad_l1[m]) for m in modalities},
        "gradient_mass_l2":  {m: _stats(all_grad_l2[m]) for m in modalities},
        "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
        "num_tasks": len(all_successes),
    }

    print(f"\n============ SUMMARY ({checkpoint_name}, n={metrics['num_tasks']}) ============")
    print(f"  Success Rate: {metrics['success_rate']:.4f}")
    for m in modalities:
        print(f"  {m:10s}  L1={metrics['gradient_mass_l1'][m]['mean']:.6f}  L2={metrics['gradient_mass_l2'][m]['mean']:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory or HF model id")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/libero_plus_grad_<ckpt_name>.json)")
    parser.add_argument("--cuda", type=int, choices=[0, 1], default=0,
                        help="CUDA device index (0 or 1)")
    args = parser.parse_args()

    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint))
    output_path = args.output or os.path.join(
        project_root, "results", f"libero_plus_grad_{ckpt_name}.json"
    )
    evaluate_gradients(args.checkpoint, output_path)


if __name__ == "__main__":
    main()