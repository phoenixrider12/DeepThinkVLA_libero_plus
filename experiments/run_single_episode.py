import os
import json
import random
import time
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import draccus
import torch
import numpy as np
from transformers import AutoProcessor

from libero.libero import benchmark
from experiments.libero_utils import (
    get_libero_env,
    save_rollout_video,
)
from experiments.deepthinkvla_utils import (
    get_vla,
)
from experiments.run_libero_plus_eval import (
    GenerateConfig,
    TaskSuite,
    set_seed_everywhere,
    validate_config,
    load_initial_states,
    run_episode,
)

@dataclass
class SingleEpisodeConfig(GenerateConfig):
    task_name: Optional[str] = None          # Name or substring of the task to evaluate
    perturbation_type: Optional[str] = None  # Perturbation category (e.g. Robot Initial States)
    episode_idx: int = 0                     # Episode index / initial state index to run (default: 0)

@draccus.wrap()
def main(cfg: SingleEpisodeConfig):
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load name to category mapping
    with open("name_to_category.json", "r") as f:
        name_to_category = json.load(f)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    def normalize_str(s):
        if s is None:
            return ""
        return s.lower().replace(" ", "_").replace("-", "_").strip()

    # Find matching task IDs
    matched_tasks = []
    target_task_norm = normalize_str(cfg.task_name)
    target_pert_norm = normalize_str(cfg.perturbation_type)

    for tid in range(task_suite.n_tasks):
        task_name = task_suite.get_task_names()[tid]
        category = name_to_category.get(task_name, None)

        task_matches = (not cfg.task_name) or (target_task_norm in normalize_str(task_name))
        pert_matches = (not cfg.perturbation_type) or (normalize_str(category) == target_pert_norm)

        if task_matches and pert_matches:
            matched_tasks.append((tid, task_name, category))

    if not matched_tasks:
        print(f"❌ Error: No tasks matched task_name='{cfg.task_name}' and perturbation_type='{cfg.perturbation_type}'")
        return

    # Print matching task list
    if len(matched_tasks) > 1:
        print(f"⚠️ Warning: Multiple tasks matched. Selecting the first match:")
        for idx, (tid, name, cat) in enumerate(matched_tasks):
            print(f"  [{idx}] ID {tid}: '{name}' (Category: {cat})")
    
    selected_task_id, selected_task_name, selected_category = matched_tasks[0]
    print(f"✨ Selected Task: ID {selected_task_id} - '{selected_task_name}' (Category: {selected_category})")

    # Load model and Components
    print("⏳ Loading model...")
    model, unomrmalize_action = get_vla(cfg)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)

    # Load initial states
    task = task_suite.get_task(selected_task_id)
    initial_states, _ = load_initial_states(cfg, task_suite, selected_task_id)

    if cfg.episode_idx < 0 or cfg.episode_idx >= len(initial_states):
        print(f"❌ Error: Invalid episode_idx={cfg.episode_idx}. Total available episodes for this task: {len(initial_states)}")
        return

    initial_state = initial_states[cfg.episode_idx]
    print(f"🚀 Running Episode Index {cfg.episode_idx}...")

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)

    # Setup trajectory saving and video rollout
    save_video = False

    # Run episode
    success, replay_images = run_episode(
        cfg=cfg,
        env=env,
        task_description=task_description,
        model=model,
        unomrmalize_action=unomrmalize_action,
        resize_size=cfg.img_resize_size,
        processor=processor,
        initial_state=initial_state,
        save_video=save_video,
        task_id=selected_task_id,
    )

    print(f"\n🎉 Episode Finished! Success: {success}")

    # Save video rollout
    if save_video:
        save_rollout_video(
            replay_images,
            success=success,
            task_description=task_description,
            episode_id=f"{selected_task_id}_ep_{cfg.episode_idx}"
        )

if __name__ == "__main__":
    main()
