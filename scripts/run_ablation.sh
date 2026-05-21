#!/bin/bash
# Run ablation: all LoRA ranks sequentially (each on 8 GPUs)

CONDA_ENV="/home/jtan/miniconda3/envs/rl_hw3"
LF_DIR="/home/jtan/llm/LLaMA-Factory"

for RANK in 8 16 32 64; do
    echo "============ Starting rank=$RANK ============"
    SESSION="cyberlllm_ablation_rank${RANK}"
    CONFIG="cyberlllm_sft_rank${RANK}.yaml"
    [ $RANK -eq 32 ] && CONFIG="cyberlllm_sft_8gpu.yaml"

    tmux new-session -d -s "$SESSION" \
        "cd $LF_DIR && \
         FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
         $CONDA_ENV/bin/llamafactory-cli train configs/$CONFIG 2>&1 | tee /home/jtan/llm_sec/logs/ablation_rank${RANK}.log; \
         echo Rank training DONE"
    echo "Launched rank=$RANK in tmux session $SESSION"
    echo "Waiting for completion before next rank..."
    tmux wait-for -S "$SESSION"
done
echo "All ablation runs complete!"
