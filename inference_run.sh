#!/bin/bash

RUN_NAME='llama32_1B_Inst'
MODEL_NAME='meta-llama/Llama-3.2-1B-Instruct'
MODEL_PROVIDER='hf'
TENSOR_PARALLEL=false
DATA_PARALLEL_SIZE=2

export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub
export SAMPLES_RESULTS_DIR=/workspace/lm_harness_runs/
export TOKENIZERS_PARALLELISM=false

NO_DP="9_0"
OK_DP="1_0 2_5"

for run_idx in $NO_DP $OK_DP; do
	if [[ $NO_DP =~ (^|[[:space:]])$run_idx($|[[:space:]]) ]]; then
		CMD="accelerate launch \
			--num_processes 1 \
			lmh_test.py \
			--run_name \"$RUN_NAME-question$run_idx\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index $run_idx \
			--inference_provider \"$MODEL_PROVIDER\""
	else
		CMD="accelerate launch \
			--num_processes $DATA_PARALLEL_SIZE \
			lmh_test.py \
			--run_name \"$RUN_NAME-question$run_idx\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index $run_idx \
			--inference_provider \"$MODEL_PROVIDER\""
	fi

	if [ "$TENSOR_PARALLEL" = true ]; then
		CMD="$CMD --parallelize"
	fi

	echo "Running: $CMD"
	eval $CMD
	sleep 5
done
