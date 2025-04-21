# ./inference_run_legacy.sh llama31_8B_Inst meta-llama/Llama-3.1-8B-Instruct 32
# sleep 5
# ./inference_run_legacy.sh Qwen25_7B_Inst Qwen/Qwen2.5-7B-Instruct 32
# sleep 5
./inference_run_legacy.sh llama31_8B meta-llama/Llama-3.1-8B 32
sleep 5
./inference_run_legacy.sh Qwen25_7B Qwen/Qwen2.5-7B 32
