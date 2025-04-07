RUN_NAME='llama32_1B_Inst'
MODEL_NAME='meta-llama/Llama-3.2-1B-Instruct'
MODEL_PROVIDER='hf'
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1

export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=/workspace/huggingface/hub
export SAMPLES_RESULTS_DIR=/workspace/lm_harness_runs/
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  lmh_test.py \
  --run_name "$RUN_NAME-question2_2" \
  --model_string "$MODEL_NAME" \
  --question_document_index 2_2 \
  --inference_provider "$MODEL_PROVIDER" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --data_parallel_size "$DATA_PARALLEL_SIZE"
sleep 10

# accelerate launch \
#   lmh_test.py \
#   --run_name "$RUN_NAME-question5_0" \
#   --model_string "$MODEL_NAME" \
#   --question_document_index 5_0 \
#   --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
#   --data_parallel_size "$DATA_PARALLEL_SIZE"
# sleep 10