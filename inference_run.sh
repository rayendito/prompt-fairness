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

# NO_DP="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
# OK_DP="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"

nodp="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
zero="0_0"
one="1_0"
two="2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14"
three="3_0"
four="4_0"
twenty="20_0"
twentyone="21_0"
twentytwo="22_0"
twentythree="23_0"
twentyfour="24_0"
twentyfive="25_0 25_1"
twentysix="26_0 26_1"
twentyseven="27_0 27_1 27_2 27_3 27_4 27_5"
twentyeight="28_0"

EXP_NAMES=(
  nodp
  zero
  one
  two
  three
  four
  twenty
  twentyone
  twentytwo
  twentythree
  twentyfour
  twentyfive
  twentysix
  twentyseven
  twentyeight
)

for name in "${EXP_NAMES[@]}"; do
	str="${!name}"  # Indirect expansion to get value of variable named $name
	if [[ "$name" == "nodp" ]]; then
		CMD="accelerate launch \
			--num_processes 1 \
			lmh_test.py \
			--run_name \"$RUN_NAME-question-$name\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index \"$str\" \
			--inference_provider \"$MODEL_PROVIDER\""
	else
		CMD="accelerate launch \
			--num_processes $DATA_PARALLEL_SIZE \
			lmh_test.py \
			--run_name \"$RUN_NAME-question-$name\" \
			--model_string \"$MODEL_NAME\" \
			--question_document_index \"$str\" \
			--inference_provider \"$MODEL_PROVIDER\""
	fi

	if [ "$TENSOR_PARALLEL" = true ]; then
		CMD="$CMD --parallelize"
	fi

	echo "Running: $CMD"
	eval $CMD
	sleep 3
done
