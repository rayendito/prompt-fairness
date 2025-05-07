set -a
source ./.env
set +a

python -m src.scripts.postprocess --run_name Qwen25_7B_Inst --inference_type "separate"
# sleep 2
# python -m src.scripts.postprocess --run_name Qwen25_1dot5B_Inst --inference_type "separate"
# sleep 2
# python -m src.scripts.postprocess --run_name llama31_8B_Inst --inference_type "separate"
# sleep 2
# python -m src.scripts.postprocess --run_name Qwen25_7B_Inst --inference_type "separate"
# sleep 2
# python -m src.scripts.postprocess --run_name llama31_70b_Inst --inference_type "separate" --recalculate
# sleep 2
# python -m src.scripts.postprocess --run_name qwen25_72b_Inst --inference_type "separate" --recalculate
