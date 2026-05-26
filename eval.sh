set -x

export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT_NAME='deepthinkvla'
export SWANLAB_API_KEY='YOUR_API_KEY'
# For open-source safety, do NOT hardcode credentials here.
# If you want SwanLab logging, set SWANLAB_API_KEY in your environment and set SWANLAB_MODE to 'cloud-only' or 'local'.
export SWANLAB_MODE='disabled' # cloud-only, local, disabled
export CUDA_VISIBLE_DEVICES=1
unset DISPLAY
export SAPIEN_RENDERER=cpu


# libero_object, libero_spatial, libero_goal, libero_10, libero_90
# To run evaluation for a specific perturbation type, add: --perturbation_type "Category Name"
# Available Categories: "Robot Initial States", "Background Textures", "Camera Viewpoints", "Objects Layout", "Language Instructions", "Light Conditions", "Sensor Noise"
# (Matching is robust, case-insensitive, and supports snake_case, e.g. "robot_initial_states" works too)

python experiments/run_libero_plus_eval.py \
    --pretrained_checkpoint checkpoints/base_cot \
    --num_images_in_input 2 \
    --task_suite_name libero_10 \
    --max_new_tokens 2048 \
    --project_name $SWANLAB_PROJECT_NAME \
    --swanlab_api_key $SWANLAB_API_KEY \
    --swanlab_mode $SWANLAB_MODE \
    --seed 429 \
    --panel_width_px 812 \
    --video_save_freq 1 \
    --task_start 600 \
    --task_end 1200 \
    # --save_traj True \
    # --perturbation_type "robot_initial_states"


# Split a run into chunks via task_start/task_end (output JSON names include the range, so they won't collide).
#   First chunk:  --task_start 0    --task_end 1200
#   Second chunk: --task_start 1200            (omit --task_end to run to the end)