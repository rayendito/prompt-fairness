export HF_HOME=/mnt/data/users/ryandito/huggingface
export HF_HUB_CACHE=/mnt/data/users/ryandito/huggingface/hub
export SAMPLES_RESULTS_DIR=/mnt/data/users/ryandito/lm_harness_runs/

# python postprocess.py --run_name llama32_1B_Inst --inference_type "separate"
# sleep 2
# python postprocess.py --run_name Qwen25_7B_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name Qwen25_1dot5B_Inst --inference_type "separate" --recalculate
# sleep 2
# python postprocess.py --run_name llama31_8B_Inst --inference_type "separate" --recalculate

python postprocess.py --run_name llama31_70b_Inst --inference_type "separate" --recalculate
sleep 2
python postprocess.py --run_name qwen25_72b_Inst --inference_type "separate" --recalculate
