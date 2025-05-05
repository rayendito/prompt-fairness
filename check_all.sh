RESDIR="/mnt/data/users/ryandito/lm_harness_runs_v2/"
# python check_runs.py --run_name llama31_8B_Inst --results_dir "$RESDIR"
# python check_runs.py --run_name Qwen25_7B_Inst --results_dir "$RESDIR"
# python check_runs.py --run_name llama32_1B_Inst --results_dir "$RESDIR"
# python check_runs.py --run_name Qwen25_1dot5B_Inst --results_dir "$RESDIR"

# python check_runs.py --run_name llama31_8B --results_dir "$RESDIR"
# python check_runs.py --run_name Qwen25_7B --results_dir "$RESDIR"


python check_runs.py --run_name gpt_4o --results_dir "$RESDIR" --through_api