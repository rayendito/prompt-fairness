export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub
export SAMPLES_RESULTS_DIR=/workspace/lm_harness_runs_v2/

# python postprocess.py --run_name llama32_1B_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name llama31_8B_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name Qwen25_1dot5B_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name Qwen25_7B_Inst --inference_type "separate" --recalculate
# sleep 2

python postprocess.py --run_name gpt_4o --inference_type "api" --recalculate

# python postprocess.py --run_name llama31_70b_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name qwen25_72b_Inst --inference_type "separate" --recalculate
