#!/bin/bash
# Launch 8-GPU SFT training in tmux
# Usage: bash run_sft.sh [rank8|rank16|rank32|rank64]

RANK=${1:-rank32}
SESSION="cyberlllm_sft_${RANK}"
CONDA_ENV="/home/jtan/miniconda3/envs/rl_hw3"
LF_DIR="/home/jtan/llm/LLaMA-Factory"

case $RANK in
    rank8)   CONFIG="cyberlllm_sft_rank8.yaml"    ;;
    rank16)  CONFIG="cyberlllm_sft_rank16.yaml"   ;;
    rank32)  CONFIG="cyberlllm_sft_8gpu.yaml"     ;;
    rank64)  CONFIG="cyberlllm_sft_rank64.yaml"   ;;
    *)       echo "Usage: $0 [rank8|rank16|rank32|rank64]"; exit 1 ;;
esac

echo "Starting SFT training: $RANK -> $CONFIG"
echo "tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd $LF_DIR && \
     FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
     $CONDA_ENV/bin/llamafactory-cli train configs/$CONFIG 2>&1 | tee /home/jtan/llm_sec/logs/sft_${RANK}.log"

echo "Training started. Monitor with: tmux attach -t $SESSION"
echo "Or check log: tail -f /home/jtan/llm_sec/logs/sft_${RANK}.log"
