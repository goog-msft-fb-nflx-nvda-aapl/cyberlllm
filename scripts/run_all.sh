#!/bin/bash
# Master script: prepare data → SFT → DPO → evaluate
# Run from /home/jtan/llm_sec/

set -e
CONDA_ENV="/home/jtan/miniconda3/envs/rl_hw3"
LF_DIR="/home/jtan/llm/LLaMA-Factory"
PROJECT_DIR="/home/jtan/llm_sec"

echo "====== Step 1: Prepare Data ======"
$CONDA_ENV/bin/python $PROJECT_DIR/data/prepare_data.py

echo ""
echo "====== Step 2: SFT Training (8 GPUs, rank=32) ======"
cd $LF_DIR
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
    $CONDA_ENV/bin/llamafactory-cli train configs/cyberlllm_sft_8gpu.yaml 2>&1 | tee $PROJECT_DIR/logs/sft_rank32.log

echo ""
echo "====== Step 3: DPO Alignment (8 GPUs) ======"
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
    $CONDA_ENV/bin/llamafactory-cli train configs/cyberlllm_dpo_8gpu.yaml 2>&1 | tee $PROJECT_DIR/logs/dpo.log

echo ""
echo "====== Step 4: Evaluation ======"
bash $PROJECT_DIR/scripts/run_eval.sh

echo ""
echo "====== All done! Check $PROJECT_DIR/reports/eval_results/ ======"
