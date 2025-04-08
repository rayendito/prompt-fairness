export SAMPLES_RESULTS_DIR=dummy_lm_harness_runs

EXPERIMENT_NAME="llama32_1B_Inst"
FULL_RUN="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"
FULL_RUN="0_0 2_5 18_0"

for run_idx in $FULL_RUN; do
    CMD="python postprocess.py --run_name $EXPERIMENT_NAME --question_document_index $run_idx"
	echo "Running: $CMD"
	eval $CMD
done
