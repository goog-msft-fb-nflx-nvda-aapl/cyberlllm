#!/bin/bash
# Launch 8-GPU DPO training after SFT is complete

SESSION="cyberlllm_dpo"
CONDA_ENV="/home/jtan/miniconda3/envs/rl_hw3"
LF_DIR="/home/jtan/llm/LLaMA-Factory"

echo "Starting DPO training..."
echo "tmux session: $SESSION"

tmux new-session -d -s "$SESSION" \
    "cd $LF_DIR && \
     FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
     $CONDA_ENV/bin/llamafactory-cli train configs/cyberlllm_dpo_8gpu.yaml 2>&1 | tee /home/jtan/llm_sec/logs/dpo.log"

echo "DPO started. Monitor: tmux attach -t $SESSION"
