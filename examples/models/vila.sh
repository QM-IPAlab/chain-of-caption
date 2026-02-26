export HF_HOME="~/.cache/huggingface"
export HF_TOKEN="your_token_here"

# --- Inference options (env vars below) ---
#   DESC_MODE       : "none" | "all" | integer for grounded description(e.g. 5) — description mode
#   DRAW_BBOX       : true | false — draw bounding boxes
#   COC             : true | false — chain-of-caption
#   CROP_AND_ZOOM   : true | false — crop and zoom
#
# --- Debug visualization ---
#   DEBUG_OUTPUT_PATH        : directory to save debug images (empty = disabled)
#   DEBUG_DRAW_TEXT         : true | false — overlay text on debug images
#   DEBUG_DRAW_BOUNDING_BOX : true | false — draw boxes on debug images

DESC_MODE="${DESC_MODE:-none}"
DRAW_BBOX="${DRAW_BBOX:-false}"
COC="${COC:-false}"
CROP_AND_ZOOM="${CROP_AND_ZOOM:-false}"
DEBUG_OUTPUT_PATH="${DEBUG_OUTPUT_PATH:-}"
DEBUG_DRAW_TEXT="${DEBUG_DRAW_TEXT:-false}"
DEBUG_DRAW_BOUNDING_BOX="${DEBUG_DRAW_BOUNDING_BOX:-false}"

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model vila \
    --model_args=pretrained=Efficient-Large-Model/NVILA-8B,desc_mode=${DESC_MODE},draw_bbox=${DRAW_BBOX},coc=${COC},crop_and_zoom=${CROP_AND_ZOOM},debug_output_path=${DEBUG_OUTPUT_PATH},debug_draw_text=${DEBUG_DRAW_TEXT},debug_draw_bounding_box=${DEBUG_DRAW_BOUNDING_BOX} \
    --tasks refcoco_bbox_rec_test \
    --output_path /data/EECS-GenH2R/yik/Projects/lmms-eval/results/ \
    --log_samples \
    --batch_size 1
