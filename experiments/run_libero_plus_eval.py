"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""
import sys
sys.path.append("./")
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# import swanlab
from transformers import AutoProcessor
import torch
import random
import time

from experiments.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.deepthinkvla_utils import (
    resize_image_for_policy,
    get_vla,
    get_vla_action,
    get_vla_action_with_steered_reasoning,
    get_vla_action_mask_cot,
    get_vla_action_mask_cot_random,
    compose_with_sidepanel,
    binarize_gripper_action
)
from experiments.reasoning_steering import steer_reasoning
from sft.constants import NUM_ACTIONS_CHUNK


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
def expand_bare_bool_flag(flag: str) -> None:
    """Let draccus bool fields be used as bare CLI flags."""
    if flag not in sys.argv:
        return

    flag_index = sys.argv.index(flag)
    has_value = flag_index + 1 < len(sys.argv) and not sys.argv[flag_index + 1].startswith("--")
    if not has_value:
        sys.argv.insert(flag_index + 1, "True")


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_MIX = 'libero_mix'


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 620,  # longest training demo has 620 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = "yinchenghust/sft_cot"     # Pretrained checkpoint path (or HF repo id)

    num_images_in_input: int = 2                # Number of images in input context

    max_new_tokens: int = 2048                       # Maximum number of cot tokens to generate (COT only)

    compute_dtype: str = "bfloat16"                    # Model compute dtype (float32, float16, bfloat16)

    img_resize_size: int = 224                     # Input image resolution for the model

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_OBJECT     # Task suite
    task_id: Optional[int] = None                     # If set, evaluate only this task index in the suite
    task_category: Optional[str] = None               # If set, evaluate only tasks matching this category
    skip_task_ids: Optional[str] = None               # Comma-separated task indices to skip, e.g. "0,3,7"
    skip_task_name_contains: Optional[str] = None     # Pipe-separated text patterns for task names to skip
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # logs_mask_cot, logs_mask_cot_random, logs

    project_name: str = "deepthinkvla"                 # Name of project to log to
    swanlab_api_key: Optional[str] = None              # Prefer env var; keep None for open-source safety
    swanlab_mode: str = 'disabled'                      # cloud-only, local, disabled

    seed: int = 429                                    # Random Seed (for reproducibility)

    panel_width_px: int = 812                         # Width of side panel for displaying CoT text

    #################################################################################################################
    # Reasoning steering / override
    #################################################################################################################
    use_reasoning_steering: bool = False             # Call steer_reasoning(image, prompt, current_reasoning)

    # fmt: on

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assert not (
        cfg.task_id is not None and cfg.task_category is not None
    ), "Set only one of task_id or task_category."

def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-deepthinkvla-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize SwanLab logging if enabled
    if cfg.swanlab_mode != "disabled":
        # Prefer environment variable; fall back to cfg
        api_key = os.environ.get("SWANLAB_API_KEY", None) or cfg.swanlab_api_key
        if api_key:
            swanlab.login(api_key)  # NOTE: previous login information will be overwritten
            swanlab.init(
                project=cfg.project_name,
                experiment_name=run_id,
                logdir=cfg.local_log_dir,
                mode=cfg.swanlab_mode,
            )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def parse_task_id_list(task_ids: Optional[str]) -> set[int]:
    if task_ids is None or task_ids.strip() == "":
        return set()
    return {int(task_id.strip()) for task_id in task_ids.split(",") if task_id.strip()}


def normalize_task_text(text: str) -> str:
    return " ".join(text.replace("_", " ").replace("-", " ").lower().split())


def text_matches(needle: str, haystack: str) -> bool:
    normalized_needle = normalize_task_text(needle)
    normalized_haystack = normalize_task_text(haystack)
    return normalized_needle in normalized_haystack or normalized_haystack in normalized_needle


def get_task_search_text(task_suite, task_id: int) -> str:
    task = task_suite.get_task(task_id)
    candidates = [task_suite.get_task_names()[task_id]]
    for attr in ("language", "description", "name"):
        value = getattr(task, attr, None)
        if isinstance(value, str):
            candidates.append(value)
    return normalize_task_text(" ".join(candidates))


def parse_task_name_patterns(patterns: Optional[str]) -> list[str]:
    if patterns is None or patterns.strip() == "":
        return []
    return [
        normalize_task_text(pattern)
        for pattern in patterns.split("|")
        if pattern.strip()
    ]


def load_task_classification():
    classification_path = Path("libero/libero/benchmark/task_classification.json")
    if not classification_path.exists():
        return None
    with open(classification_path, "r") as f:
        return json.load(f)


def collect_texts(node) -> list[str]:
    texts = []
    if isinstance(node, str):
        texts.append(node)
    elif isinstance(node, list):
        for item in node:
            texts.extend(collect_texts(item))
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                texts.append(key)
            texts.extend(collect_texts(value))
    return texts


def classification_contains_task(node, task_text: str) -> bool:
    return any(text_matches(text, task_text) for text in collect_texts(node))


def task_matches_category(task_classification, task_text: str, category: str) -> bool:
    """Match task/category in common classification JSON layouts."""
    if task_classification is None:
        return False

    def visit(node, category_seen: bool = False, task_seen: bool = False) -> bool:
        if isinstance(node, str):
            return (category_seen or text_matches(category, node)) and (task_seen or text_matches(task_text, node))

        if isinstance(node, list):
            node_texts = collect_texts(node)
            node_has_category = category_seen or any(text_matches(category, text) for text in node_texts)
            node_has_task = task_seen or any(text_matches(text, task_text) for text in node_texts)
            if node_has_category and node_has_task:
                return True
            return any(visit(item, node_has_category, node_has_task) for item in node)

        if isinstance(node, dict):
            for key, value in node.items():
                key_has_category = isinstance(key, str) and text_matches(category, key)
                key_has_task = isinstance(key, str) and text_matches(key, task_text)
                next_category_seen = category_seen or key_has_category
                next_task_seen = task_seen or key_has_task

                # category -> subtree containing task(s)
                if next_category_seen and classification_contains_task(value, task_text):
                    return True

                # task -> subtree containing category/category metadata
                if next_task_seen and any(text_matches(category, text) for text in collect_texts(value)):
                    return True

                if visit(value, next_category_seen, next_task_seen):
                    return True

        return False

    return visit(task_classification)


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    unomrmalize_action,
    resize_size,
    processor=None,
    initial_state=None,
    save_video=False,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)

    # Setup
    t = 0
    replay_images = []
    cot_replay = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action())
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                with torch.no_grad():
                    # mask_cot: get_vla_action_mask_cot
                    # mask_cot_random: get_vla_action_mask_cot_random
                    if cfg.use_reasoning_steering:
                        actions, cot_text = get_vla_action_with_steered_reasoning(
                            cfg=cfg,
                            vla=model,
                            unomrmalize_action=unomrmalize_action,
                            processor=processor,
                            obs=observation,
                            task_label=task_description,
                            reasoning_steering_fn=steer_reasoning,
                        )
                    else:
                        actions, cot_text = get_vla_action(
                            cfg=cfg,
                            vla=model,
                            unomrmalize_action = unomrmalize_action,
                            processor=processor,
                            obs=observation,
                            task_label=task_description,
                        )
                action_queue.extend(actions)
                cot_replay.append(cot_text)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = binarize_gripper_action(action)
            if save_video:
                replay_images[-1] = compose_with_sidepanel(replay_images[-1], cot_replay[-1] if len(cot_replay) > 0 else None, panel_width_px=cfg.panel_width_px)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    unomrmalize_action,
    resize_size,
    processor=None,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)

    log_message(f"\nTask: {task_description}", log_file)

    # Handle initial state
    if cfg.initial_states_path == "DEFAULT":
        # Use default initial state
        initial_state = initial_states[0]
    else:
        raise('now is not supported')

    save_video = (task_id % 25 == 0)                                    # saving video every 25 runs

    # Run episode
    success, replay_images = run_episode(
        cfg,
        env,
        task_description,
        model,
        unomrmalize_action,
        resize_size,
        processor,
        initial_state,
        save_video,
        log_file,
    )

    if save_video:
        save_rollout_video(
            replay_images, success=success, task_description=task_description, log_file=log_file, episode_id=task_id
        )

    # Log results
    log_message(f"Success: {success}", log_file)

    return success


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    with open("name_to_category.json", "r") as f:
        name_to_category =  json.load(f)
    task_classification = load_task_classification()

    ##########################################################################################
    # Initialize model and components
    model, unomrmalize_action = get_vla(cfg)

    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)
    ##########################################################################################

    ##########################################################################################
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    if cfg.use_reasoning_steering:
        log_message(
            "Reasoning steering enabled. Actions will be decoded from the steered reasoning trace.",
            log_file,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = min(task_suite.n_tasks, 500)

    # Start evaluation
    result_success_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }
    result_fail_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }
    if cfg.task_category is not None:
        assert cfg.task_category in result_success_dict, (
            f"Invalid task_category {cfg.task_category!r}; expected one of {list(result_success_dict.keys())}"
        )

    if cfg.task_id is not None:
        assert 0 <= cfg.task_id < num_tasks, f"Invalid task_id {cfg.task_id}; expected 0 <= task_id < {num_tasks}"
        task_ids = [cfg.task_id]
    elif cfg.task_category is not None:
        assert task_classification is not None, "task_category requires libero/libero/benchmark/task_classification.json"
        task_ids = [
            task_id
            for task_id in range(num_tasks)
            if task_matches_category(
                task_classification,
                get_task_search_text(task_suite, task_id),
                cfg.task_category,
            )
        ]
        assert len(task_ids) > 0, f"No tasks found for category {cfg.task_category!r}"
    else:
        task_ids = range(num_tasks)

    skipped_task_ids = parse_task_id_list(cfg.skip_task_ids)
    if skipped_task_ids:
        invalid_skips = [task_id for task_id in skipped_task_ids if task_id < 0 or task_id >= num_tasks]
        assert not invalid_skips, f"Invalid skip_task_ids {invalid_skips}; expected 0 <= task_id < {num_tasks}"
        task_ids = [task_id for task_id in task_ids if task_id not in skipped_task_ids]
        assert len(task_ids) > 0, "No tasks left after applying skip_task_ids."

    if cfg.skip_task_name_contains is not None:
        skip_texts = parse_task_name_patterns(cfg.skip_task_name_contains)
        task_ids = [
            task_id
            for task_id in task_ids
            if not any(skip_text in get_task_search_text(task_suite, task_id) for skip_text in skip_texts)
        ]
        assert len(task_ids) > 0, "No tasks left after applying skip_task_name_contains."

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    if cfg.task_id is not None:
        log_message(f"Evaluating only task_id: {cfg.task_id}", log_file)
    if cfg.task_category is not None:
        log_message(f"Evaluating only task_category: {cfg.task_category} ({len(task_ids)} tasks)", log_file)
    if skipped_task_ids:
        log_message(f"Skipping task_ids: {sorted(skipped_task_ids)}", log_file)
    if cfg.skip_task_name_contains is not None:
        log_message(f"Skipping task names containing any of: {parse_task_name_patterns(cfg.skip_task_name_contains)}", log_file)

    total_successes = 0
    checkpoint_name = Path(cfg.pretrained_checkpoint).name
    for task_id in tqdm.tqdm(task_ids):
        # task_id = 0
        success = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            unomrmalize_action,
            cfg.img_resize_size,
            processor,
            log_file,
        )
        result_success_dict[name_to_category[task_suite.get_task_names()[task_id]]] += 1 if success else 0
        result_fail_dict[name_to_category[task_suite.get_task_names()[task_id]]] += 1 if not success else 0
        if success:
            total_successes += 1
            
        if (task_id + 1) % 10 == 0:
            with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_success_outcome.json", "w") as f:
                json.dump(result_success_dict, f, indent=4)
            with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_fail_outcome.json", "w") as f:
                json.dump(result_fail_dict, f, indent=4)

    # Calculate final success rate
    num_eval_tasks = len(task_ids) if isinstance(task_ids, list) else num_tasks
    final_success_rate = float(total_successes) / float(num_eval_tasks)

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {num_eval_tasks}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_success_outcome.json", "w") as f:
        json.dump(result_success_dict, f, indent=4)
    with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_fail_outcome.json", "w") as f:
        json.dump(result_fail_dict, f, indent=4)
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    expand_bare_bool_flag("--use_reasoning_steering")
    eval_libero()
