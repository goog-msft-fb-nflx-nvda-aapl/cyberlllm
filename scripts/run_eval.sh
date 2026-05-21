#!/bin/bash
# Evaluate all model checkpoints on MMLU computer_security
# Usage: bash run_eval.sh

CONDA_ENV="/home/jtan/miniconda3/envs/rl_hw3"
LF_DIR="/home/jtan/llm/LLaMA-Factory"
EVAL_DIR="/home/jtan/llm_sec/eval"
OUT_DIR="/home/jtan/llm_sec/reports/eval_results"
MODEL="/home/jtan/llm/models/Qwen2.5-7B-Instruct"
mkdir -p $OUT_DIR

echo "=== Evaluating Baseline (no fine-tuning) ==="
$CONDA_ENV/bin/python $EVAL_DIR/evaluate.py \
    --model_path $MODEL \
    --output $OUT_DIR/baseline.json

echo "=== Evaluating SFT rank=32 ==="
$CONDA_ENV/bin/python $EVAL_DIR/evaluate.py \
    --model_path $MODEL \
    --adapter_path $LF_DIR/saves/cyberlllm_sft_8gpu \
    --output $OUT_DIR/sft_rank32.json

echo "=== Evaluating DPO ==="
$CONDA_ENV/bin/python $EVAL_DIR/evaluate.py \
    --model_path $MODEL \
    --adapter_path $LF_DIR/saves/cyberlllm_dpo_8gpu \
    --output $OUT_DIR/dpo.json

# Print summary
$CONDA_ENV/bin/python << PYEOF
import json, glob
print("\n=== MMLU Computer Security Results ===")
for f in sorted(glob.glob("$OUT_DIR/*.json")):
    d = json.load(open(f))
    name = f.split("/")[-1].replace(".json","")
    acc = d.get("accuracy", 0)
    correct = d.get("correct", 0)
    total = d.get("num_samples", 0)
    print(f"  {name:20s}: {acc*100:.1f}% ({correct}/{total})")
PYEOF
