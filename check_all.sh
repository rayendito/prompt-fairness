RESDIR="/l/users/ryandito.diandaru/lm_harness_runs_v3"

echo "llama32_1B_Inst"
python check_runs.py --run_name llama32_1B_Inst --results_dir "$RESDIR"
echo "llama31_8B_Inst"
python check_runs.py --run_name llama31_8B_Inst --results_dir "$RESDIR"

echo "Qwen25_1dot5B_Inst"
python check_runs.py --run_name Qwen25_1dot5B_Inst --results_dir "$RESDIR"
echo "Qwen25_7B_Inst"
python check_runs.py --run_name Qwen25_7B_Inst --results_dir "$RESDIR"

# python check_runs.py --run_name gpt_4o --results_dir "$RESDIR" --through_api
# python check_runs.py --run_name Qwen25_72B_Inst --results_dir "$RESDIR" --through_api
# python check_runs.py --run_name llama31_70B_Inst --results_dir "$RESDIR" --through_api