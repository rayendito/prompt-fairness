# full runs
# NO_DP="5_0 6_0 7_0 8_0 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0 17_0 18_0 19_0"
# OK_DP="0_0 1_0 4_0 20_0 26_0 26_1 2_0 2_1 2_2 2_3 2_4 2_5 2_6 2_7 2_8 2_9 2_10 2_11 2_12 2_13 2_14 3_0 21_0 22_0 23_0 24_0 25_0 25_1 27_0 27_1 27_2 27_3 27_4 27_5 28_0"

# RUN_NAME="gpt_4o"
# MODEL_PROVIDER="openai"
# MODEL_NAME="gpt-4o"
# NO_DP="24_0"
# OK_DP=""

RUN_NAME="Qwen25_72B_Inst"
MODEL_PROVIDER="openrouter"
MODEL_NAME="qwen/qwen-2.5-72b-instruct"
NO_DP=""
OK_DP="2_13"

# RUN_NAME="llama31_70B_Inst"
# MODEL_PROVIDER="openrouter"
# MODEL_NAME="meta-llama/llama-3.1-70b-instruct"
# NO_DP=""
# OK_DP="20_0 26_0 24_0 25_0"

for task in $NO_DP $OK_DP; do
    SESSION_NAME="$RUN_NAME-$task"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        CMD="tmux new-session -d -s \"$SESSION_NAME\" './inference_run_oai.sh $RUN_NAME $MODEL_PROVIDER $MODEL_NAME $task > \"api_inference.log/$SESSION_NAME.log\" 2>&1'"
        echo "$CMD"
        eval $CMD
    else
        echo "Session $SESSION_NAME already exists. Skipping."
    fi
done
