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

# Set TASK_ID to evaluate one task from the selected suite, e.g.:
#   TASK_ID=3 bash eval.sh
TASK_ID_ARG=()
if [[ -n "${TASK_ID:-}" ]]; then
    TASK_ID_ARG=(--task_id "$TASK_ID")
fi

# Set TASK_CATEGORY to evaluate one LIBERO+ variation category, e.g.:
#   TASK_CATEGORY="Camera Viewpoints" bash eval.sh
TASK_CATEGORY_ARG=()
if [[ -n "${TASK_CATEGORY:-}" ]]; then
    TASK_CATEGORY_ARG=(--task_category "$TASK_CATEGORY")
fi

# Set SKIP_TASK_IDS to skip task indices, e.g.:
#   SKIP_TASK_IDS=0 bash eval.sh
SKIP_TASK_IDS_ARG=()
if [[ -n "${SKIP_TASK_IDS:-}" ]]; then
    SKIP_TASK_IDS_ARG=(--skip_task_ids "$SKIP_TASK_IDS")
fi

# Set SKIP_TASK_NAME_CONTAINS to skip every task family matching text.
# Separate multiple patterns with "|", e.g.:
#   SKIP_TASK_NAME_CONTAINS="turn on the stove and put the moka pot on it" bash eval.sh
#   SKIP_TASK_NAME_CONTAINS="turn on the stove and put the moka pot on it|put the bowl on the plate" bash eval.sh
SKIP_TASK_NAME_ARG=()
if [[ -n "${SKIP_TASK_NAME_CONTAINS:-}" ]]; then
    SKIP_TASK_NAME_ARG=(--skip_task_name_contains "$SKIP_TASK_NAME_CONTAINS")
fi

# libero_object, libero_spatial, libero_goal, libero_10, libero_90

python experiments/run_libero_plus_eval.py \
    --num_images_in_input 2 \
    --task_suite_name libero_10 \
    "${TASK_ID_ARG[@]}" \
    "${TASK_CATEGORY_ARG[@]}" \
    "${SKIP_TASK_IDS_ARG[@]}" \
    "${SKIP_TASK_NAME_ARG[@]}" \
    --max_new_tokens 2048 \
    --project_name $SWANLAB_PROJECT_NAME \
    --swanlab_api_key $SWANLAB_API_KEY \
    --swanlab_mode $SWANLAB_MODE \
    --seed 429 \
    --panel_width_px 812 \
    --pretrained_checkpoint /data/aryaman/DeepThinkVLA_libero_plus/checkpoints/grad_loss_cot
    # --use_reasoning_steering
