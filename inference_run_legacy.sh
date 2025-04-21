#!/bin/bash

RUN_NAME="$1"
MODEL_NAME="$2"
MODEL_PROVIDER='hf'
TENSOR_PARALLEL=false
DATA_PARALLEL_SIZE=1
BATCH_SIZE=$3

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/data/users/ryandito/huggingface
export HF_HUB_CACHE=/mnt/data/users/ryandito/huggingface/hub
export SAMPLES_RESULTS_DIR=/mnt/data/users/ryandito/lm_harness_runs/
export TOKENIZERS_PARALLELISM=false

NO_DP="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
OK_DP="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"

for run_idx in $NO_DP $OK_DP; do
	if [[ $NO_DP =~ (^|[[:space:]])$run_idx($|[[:space:]]) ]]; then
		CMD="accelerate launch \
			--num_processes 1 \
			lmh_test.py \
			--run_name \"$RUN_NAME-question$run_idx\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index \"$run_idx\" \
			--inference_provider \"$MODEL_PROVIDER\" \
			--batch_size $BATCH_SIZE"
	else
		CMD="accelerate launch \
			--num_processes $DATA_PARALLEL_SIZE \
			lmh_test.py \
			--run_name \"$RUN_NAME-question$run_idx\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index \"$run_idx\" \
			--inference_provider \"$MODEL_PROVIDER\" \
			--batch_size $BATCH_SIZE"
	fi

	if [ "$TENSOR_PARALLEL" = true ]; then
		CMD="$CMD --parallelize"
	fi

	echo "Running: $CMD"
	eval $CMD
	sleep 5
done
