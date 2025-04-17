export SAMPLES_RESULTS_DIR=/l/users/ryandito.diandaru/lm_harness_runs/
EXPERIMENT_NAME="llama32_1B_Inst"

python postprocess.py --run_name $EXPERIMENT_NAME --inference_type "separate"

# for run_idx in $FULL_RUN; do
#     CMD="python postprocess.py --run_name $EXPERIMENT_NAME --question_document_index $run_idx"
# 	echo "Running: $CMD"
# 	eval $CMD
# done
