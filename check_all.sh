RESDIR="/mnt/data/users/ryandito/lm_harness_runs_v2/"

# python check_runs.py --run_name llama31_8B_Inst --results_dir "$RESDIR"
# python check_runs.py --run_name Qwen25_1dot5B_Inst --results_dir "$RESDIR"

python check_runs.py --run_name gpt_4o --results_dir "$RESDIR" --through_api
python check_runs.py --run_name Qwen25_72B_Inst --results_dir "$RESDIR" --through_api
python check_runs.py --run_name llama31_70B_Inst --results_dir "$RESDIR" --through_api