RESDIR="/mnt/data/users/ryandito/lm_harness_runs_v2/"

# python -m src.scripts.check_runs --run_name llama31_8B_Inst --results_dir "$RESDIR"
# python -m src.scripts.check_runs --run_name Qwen25_1dot5B_Inst --results_dir "$RESDIR"
python -m src.scripts.check_runs --run_name Qwen25_72B_Inst --results_dir "$RESDIR" --through_api
python -m src.scripts.check_runs --run_name llama31_70B_Inst --results_dir "$RESDIR" --through_api
python -m src.scripts.check_runs --run_name gpt_4o --results_dir "$RESDIR" --through_api