RUN_NAME="$1"
MODEL_PROVIDER="$2"
MODEL_NAME="$3"
RESULTS_DIR="/mnt/data/users/ryandito/lm_harness_runs_v2/$RUN_NAME"
REFERENCE_DIR="/mnt/data/users/ryandito/lm_harness_runs/llama32_1B_Inst"

NO_DP="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
OK_DP="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"

for run_idx in $NO_DP $OK_DP; do
    CMD="python -m src.main_experiment_runner.oai_test \
            --run_name $RUN_NAME \
            --question_document_index $run_idx \
            --model_provider $MODEL_PROVIDER \
            --model_name $MODEL_NAME \
            --results_dir $RESULTS_DIR \
            --reference_dir $REFERENCE_DIR"

	echo "Running: $CMD"
	eval $CMD
	sleep 2
done
