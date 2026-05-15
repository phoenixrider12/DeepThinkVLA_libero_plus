"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""
import sys
sys.path.append("./")
import json
import importlib
import inspect
import logging
import os
import pkgutil
import shutil
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
from libero.libero import get_libero_path

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
    get_vla_action_mask_cot,
    get_vla_action_mask_cot_random,
    compose_with_sidepanel,
    binarize_gripper_action
)
from sft.constants import NUM_ACTIONS_CHUNK


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_MIX = 'libero_mix'
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 620,  # longest training demo has 620 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_GOAL_TEMP: 300,
    TaskSuite.LIBERO_SPATIAL_TEMP: 220,
    TaskSuite.LIBERO_10_TEMP: 520,
    TaskSuite.LIBERO_OBJECT_TEMP: 280,
    TaskSuite.LIBERO_GOAL_LAN: 300,
    TaskSuite.LIBERO_SPATIAL_LAN: 220,
    TaskSuite.LIBERO_10_LAN: 520,
    TaskSuite.LIBERO_OBJECT_LAN: 280,
    TaskSuite.LIBERO_GOAL_OBJECT: 300,
    TaskSuite.LIBERO_SPATIAL_OBJECT: 220,
    TaskSuite.LIBERO_10_OBJECT: 520,
    TaskSuite.LIBERO_OBJECT_OBJECT: 280,
    TaskSuite.LIBERO_GOAL_SWAP: 300,
    TaskSuite.LIBERO_SPATIAL_SWAP: 220,
    TaskSuite.LIBERO_10_SWAP: 520,
    TaskSuite.LIBERO_OBJECT_SWAP: 280,
    TaskSuite.LIBERO_GOAL_TASK: 300,
    TaskSuite.LIBERO_SPATIAL_TASK: 220,
    TaskSuite.LIBERO_10_TASK: 520,
    TaskSuite.LIBERO_OBJECT_TASK: 280,
    TaskSuite.LIBERO_GOAL_ENV: 300,
    TaskSuite.LIBERO_SPATIAL_ENV: 220,
    TaskSuite.LIBERO_10_ENV: 520,
    TaskSuite.LIBERO_OBJECT_ENV: 280,
}

LIBERO_PRO_PERTURBATION_CATEGORIES = {
    "lan": "Semantic Perturbation",
    "object": "Object Perturbation",
    "swap": "Position Perturbation",
    "task": "Task Perturbation",
    "env": "Environment Perturbation",
}

LIBERO_PRO_SUITE_NAMES = {
    f"{base_suite}_{perturbation}"
    for perturbation in LIBERO_PRO_PERTURBATION_CATEGORIES
    for base_suite in ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
}

LIBERO_PRO_RESULT_CATEGORIES = list(LIBERO_PRO_PERTURBATION_CATEGORIES.values())


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
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    libero_pro_dataset_dir: Optional[str] = None      # Path to LIBERO-Pro-dataset with bddl_files/ and init_files/
    link_libero_pro_assets: bool = True               # Symlink Pro bddl/init folders into this LIBERO checkout if missing
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
    benchmark_dict = benchmark.get_benchmark_dict()
    assert cfg.task_suite_name in benchmark_dict, (
        f"Invalid task suite: {cfg.task_suite_name}. "
        f"Available suites include: {', '.join(sorted(benchmark_dict.keys()))}"
    )


def is_libero_pro_suite(task_suite_name: str) -> bool:
    """Return True for LIBERO-Pro perturbation suites."""
    return task_suite_name in LIBERO_PRO_SUITE_NAMES


def get_libero_pro_category(task_suite_name: str) -> Optional[str]:
    """Map a LIBERO-Pro suite suffix to the perturbation category from the dataset card."""
    return LIBERO_PRO_PERTURBATION_CATEGORIES.get(task_suite_name.rsplit("_", 1)[-1])


def ensure_libero_pro_assets(cfg: GenerateConfig, log_file=None) -> None:
    """Make LIBERO-Pro bddl/init files visible to this vendored LIBERO checkout."""
    if not is_libero_pro_suite(cfg.task_suite_name):
        return

    bddl_suite_dir = Path(get_libero_path("bddl_files")) / cfg.task_suite_name
    init_suite_dir = Path(get_libero_path("init_states")) / cfg.task_suite_name

    if bddl_suite_dir.exists() and init_suite_dir.exists():
        if cfg.libero_pro_dataset_dir is not None:
            dataset_dir = Path(cfg.libero_pro_dataset_dir).expanduser().resolve()
            ensure_libero_pro_framework_assets(dataset_dir, cfg.link_libero_pro_assets, log_file)
        return

    if cfg.libero_pro_dataset_dir is None:
        missing = []
        if not bddl_suite_dir.exists():
            missing.append(str(bddl_suite_dir))
        if not init_suite_dir.exists():
            missing.append(str(init_suite_dir))
        raise FileNotFoundError(
            "LIBERO-Pro assets are missing. Copy/symlink the dataset-card folders into this LIBERO checkout "
            f"or pass --libero_pro_dataset_dir /path/to/LIBERO-Pro-dataset. Missing: {missing}"
        )

    dataset_dir = Path(cfg.libero_pro_dataset_dir).expanduser().resolve()
    source_dirs = {
        "bddl_files": dataset_dir / "bddl_files" / cfg.task_suite_name,
        "init_files": dataset_dir / "init_files" / cfg.task_suite_name,
    }
    target_dirs = {
        "bddl_files": bddl_suite_dir,
        "init_files": init_suite_dir,
    }

    for asset_kind, source_dir in source_dirs.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing LIBERO-Pro {asset_kind} suite folder: {source_dir}")

        target_dir = target_dirs[asset_kind]
        if target_dir.exists():
            continue

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if cfg.link_libero_pro_assets:
            os.symlink(source_dir, target_dir, target_is_directory=True)
            log_message(f"Linked LIBERO-Pro {asset_kind}: {target_dir} -> {source_dir}", log_file)
        else:
            shutil.copytree(source_dir, target_dir)
            log_message(f"Copied LIBERO-Pro {asset_kind}: {source_dir} -> {target_dir}", log_file)

    ensure_libero_pro_framework_assets(dataset_dir, cfg.link_libero_pro_assets, log_file)


def ensure_libero_pro_framework_assets(
    dataset_dir: Path,
    use_symlinks: bool,
    log_file=None,
) -> None:
    """Mirror LIBERO-Pro framework assets referenced by Pro BDDL files."""
    libero_pro_repo_dir = dataset_dir.parent
    source_assets_dir = libero_pro_repo_dir / "libero" / "libero" / "assets"
    target_assets_dir = Path(get_libero_path("assets"))

    if not source_assets_dir.exists():
        log_message(
            f"LIBERO-Pro framework assets not found at {source_assets_dir}; "
            "skipping optional asset mirroring.",
            log_file,
        )
        return

    for source_path in source_assets_dir.rglob("*"):
        if source_path.is_dir():
            continue

        target_path = target_assets_dir / source_path.relative_to(source_assets_dir)
        if target_path.exists():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if use_symlinks:
            os.symlink(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)

    log_message(f"Checked LIBERO-Pro framework assets from {source_assets_dir}", log_file)
    ensure_libero_pro_object_registry(libero_pro_repo_dir, use_symlinks, log_file)


def ensure_libero_pro_object_registry(
    libero_pro_repo_dir: Path,
    use_symlinks: bool,
    log_file=None,
) -> None:
    """Mirror LIBERO-Pro object registry code for Pro-only fixture categories."""
    source_objects_dir = libero_pro_repo_dir / "libero" / "libero" / "envs" / "objects"
    target_objects_dir = Path(__file__).resolve().parents[1] / "libero" / "libero" / "envs" / "objects"

    if not source_objects_dir.exists():
        log_message(
            f"LIBERO-Pro object registry not found at {source_objects_dir}; "
            "skipping optional object-code mirroring.",
            log_file,
        )
        return

    for source_path in source_objects_dir.rglob("*.py"):
        target_path = target_objects_dir / source_path.relative_to(source_objects_dir)

        if target_path.exists() and target_path.read_bytes() == source_path.read_bytes():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() or not use_symlinks:
            shutil.copy2(source_path, target_path)
        else:
            os.symlink(source_path, target_path)

    importlib.invalidate_caches()
    objects_module = sys.modules.get("libero.libero.envs.objects")
    if objects_module is not None:
        importlib.reload(objects_module)

    register_libero_pro_object_aliases(log_file)
    log_message(f"Checked LIBERO-Pro object registry from {source_objects_dir}", log_file)


def _normalize_registry_name(name: str) -> str:
    return name.lower().replace("_", "")


def register_libero_pro_object_aliases(log_file=None) -> None:
    """Register Pro object names that older LIBERO registries may omit."""
    objects_module = importlib.import_module("libero.libero.envs.objects")
    objects_dict = getattr(objects_module, "OBJECTS_DICT", None)
    if not isinstance(objects_dict, dict):
        return

    if "yellow_cabinet" in objects_dict:
        return

    for _, module_name, _ in pkgutil.walk_packages(objects_module.__path__, objects_module.__name__ + "."):
        module = importlib.import_module(module_name)
        for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
            normalized_name = _normalize_registry_name(attr_name)
            if normalized_name in {"yellowcabinet", "yellowcabinetobject"}:
                objects_dict["yellow_cabinet"] = attr_value
                log_message("Registered LIBERO-Pro object alias: yellow_cabinet", log_file)
                return

    for fallback_name in ["wooden_cabinet", "cabinet"]:
        if fallback_name in objects_dict:
            objects_dict["yellow_cabinet"] = objects_dict[fallback_name]
            log_message(
                f"Registered LIBERO-Pro object fallback: yellow_cabinet -> {fallback_name}",
                log_file,
            )
            return


def make_result_dict(cfg: GenerateConfig) -> dict:
    """Create a result counter matching LIBERO+ or LIBERO-Pro reporting."""
    if is_libero_pro_suite(cfg.task_suite_name):
        return {category: 0 for category in LIBERO_PRO_RESULT_CATEGORIES}
    return {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }


def get_result_category(cfg: GenerateConfig, task_name: str, name_to_category: dict) -> str:
    """Get the reporting bucket for one evaluated task."""
    if is_libero_pro_suite(cfg.task_suite_name):
        category = get_libero_pro_category(cfg.task_suite_name)
        if category is None:
            raise KeyError(f"No LIBERO-Pro category mapping for suite: {cfg.task_suite_name}")
        return category
    return name_to_category[task_name]




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

    ##########################################################################################
    # Initialize model and components
    model, unomrmalize_action = get_vla(cfg)

    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)
    ##########################################################################################

    ##########################################################################################
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    ensure_libero_pro_assets(cfg, log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = min(task_suite.n_tasks, 500)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    result_success_dict = make_result_dict(cfg)
    result_fail_dict = make_result_dict(cfg)
    total_successes = 0
    checkpoint_name = Path(cfg.pretrained_checkpoint).name
    for task_id in tqdm.tqdm(range(num_tasks)):
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
        result_category = get_result_category(cfg, task_suite.get_task_names()[task_id], name_to_category)
        result_success_dict[result_category] += 1 if success else 0
        result_fail_dict[result_category] += 1 if not success else 0
        if success:
            total_successes += 1
            
        if (task_id + 1) % 10 == 0:
            with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_success_outcome.json", "w") as f:
                json.dump(result_success_dict, f, indent=4)
            with open(f"{cfg.task_suite_name.lower()}_{checkpoint_name}_fail_outcome.json", "w") as f:
                json.dump(result_fail_dict, f, indent=4)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_tasks)

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {num_tasks}", log_file)
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
    eval_libero()
