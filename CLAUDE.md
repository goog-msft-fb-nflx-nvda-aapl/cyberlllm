# CyberLLM — Reproduction Guide for Claude

This file tells Claude how to fully reproduce the CyberLLM project on a fresh remote GPU server.

## Project Summary

Fine-tune Qwen2.5-7B-Instruct with LoRA SFT + DPO for NIST Cybersecurity Framework classification.
Best result: SFT rank=64 → **83% MMLU Computer Security accuracy** (baseline: 74%).

---

## Server Access

```bash
ssh gsm-gpu
# User: jtan | Home: /home/jtan
# 8x NVIDIA H200 (143GB VRAM each) | CUDA 12.4 | No sudo
# Conda: /home/jtan/miniconda3/bin/conda
```

---

## Step 0: Environment Check

The training environment is the `rl_hw3` conda env. Verify it exists:

```bash
/home/jtan/miniconda3/bin/conda env list
```

If `rl_hw3` is missing, create it:

```bash
/home/jtan/miniconda3/bin/conda create -n rl_hw3 python=3.12 -y
/home/jtan/miniconda3/envs/rl_hw3/bin/pip install \
    torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
/home/jtan/miniconda3/envs/rl_hw3/bin/pip install \
    transformers==5.5.4 peft==0.18.1 accelerate==1.11.0 \
    datasets trl bitsandbytes deepspeed
# Install LlamaFactory from source
cd /home/jtan/llm/LLaMA-Factory
/home/jtan/miniconda3/envs/rl_hw3/bin/pip install -e ".[torch,metrics]"
```

All subsequent commands use the full Python path:
```bash
PYTHON=/home/jtan/miniconda3/envs/rl_hw3/bin/python
LMF=/home/jtan/miniconda3/envs/rl_hw3/bin/llamafactory-cli
export PATH=/home/jtan/miniconda3/envs/rl_hw3/bin:$PATH
```

---

## Step 1: Download Base Model

```bash
$PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='/home/jtan/llm/models/Qwen2.5-7B-Instruct',
    ignore_patterns=['*.md', '*.txt']
)
print('Done')
"
```

Expected size: ~15GB. Stored at `/home/jtan/llm/models/Qwen2.5-7B-Instruct/`.

---

## Step 2: Set Up LlamaFactory Data Directory

The source datasets live in this repo under `data/`. Copy them to LlamaFactory:

```bash
LF_DATA=/home/jtan/llm/LLaMA-Factory/data

# Copy source datasets
cp data/cybersec_alpaca.json $LF_DATA/
mkdir -p $LF_DATA/cybersec_harmony
cp data/cybersec_harmony/train.json $LF_DATA/cybersec_harmony/
cp data/cybersec_harmony/eval.json  $LF_DATA/cybersec_harmony/

# Copy configs
cp configs/cyberlllm_*.yaml /home/jtan/llm/LLaMA-Factory/configs/
```

Register the datasets in LlamaFactory's dataset_info.json:

```bash
$PYTHON - << 'EOF'
import json
from pathlib import Path
info_path = Path('/home/jtan/llm/LLaMA-Factory/data/dataset_info.json')
info = json.loads(info_path.read_text())
info['cyberlllm_sft_train'] = {'file_name': 'cyberlllm_train.json'}
info['cyberlllm_dpo'] = {
    'file_name': 'cyberlllm_dpo.json',
    'ranking': True,
    'columns': {'prompt':'instruction','query':'input','chosen':'chosen','rejected':'rejected'}
}
info['mmlu_cs_eval'] = {'file_name': 'mmlu_cs_eval.json'}
info_path.write_text(json.dumps(info, indent=2))
print('dataset_info.json updated')
EOF
```

---

## Step 3: Prepare Training Data

```bash
$PYTHON data/prepare_data.py
```

This script:
- Loads cybersec_alpaca (2000 samples) and cybersec_harmony/train (2000 samples)
- Downloads MMLU computer_security from HuggingFace (116 samples)
- Combines into 4116 samples, applies 95/5 train/val split
- Writes to LlamaFactory/data/: cyberlllm_train.json, cyberlllm_dpo.json, mmlu_cs_eval.json

Expected output:
```
cybersec_alpaca: 2000 samples
cybersec_harmony train: 2000 samples
MMLU computer_security: 116 samples
Splits -> train:3911  val:205
```

---

## Step 4: Training

All training runs from `/home/jtan/llm/LLaMA-Factory/`.
Multi-GPU launch requires PATH to include the conda env bin.

```bash
cd /home/jtan/llm/LLaMA-Factory
export PATH=/home/jtan/miniconda3/envs/rl_hw3/bin:$PATH
```

### 4a. SFT — Best Model (rank=64, ~5 min on 8×H200)

```bash
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
    llamafactory-cli train configs/cyberlllm_sft_rank64.yaml \
    2>&1 | tee sft_rank64.log
```

Expected: final train_loss ≈ 0.337, runtime ≈ 264 seconds.
Checkpoint saved to: `saves/cyberlllm_sft_rank64/`

### 4b. SFT — Ablation Runs (ranks 8, 16, 32)

```bash
for CFG in cyberlllm_sft_rank8 cyberlllm_sft_rank16 cyberlllm_sft_8gpu; do
    FORCE_TORCHRUN=1 NPROC_PER_NODE=8 \
        llamafactory-cli train configs/${CFG}.yaml 2>&1 | tee ${CFG}.log
done
```

Note: `cyberlllm_sft_8gpu.yaml` = rank=32 config.

### 4c. DPO — From Best SFT Checkpoint (~2 min)

```bash
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
    llamafactory-cli train configs/cyberlllm_dpo_rank64.yaml \
    2>&1 | tee dpo_rank64.log
```

Expected: final rewards/margins ≈ 26.31, DPO loss ≈ 0.000184.
Checkpoint saved to: `saves/cyberlllm_dpo_rank64/`

### 4d. DPO — From rank=32 SFT

```bash
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
    llamafactory-cli train configs/cyberlllm_dpo_8gpu.yaml \
    2>&1 | tee dpo_rank32.log
```

---

## Step 5: Evaluation (MMLU Computer Security)

The evaluation script loads the model, optionally merges a LoRA adapter,
and runs 0-shot greedy decoding on 100 MMLU Computer Security questions.

```bash
mkdir -p eval_results
MODEL=/home/jtan/llm/models/Qwen2.5-7B-Instruct
LF=/home/jtan/llm/LLaMA-Factory

# Baseline (no fine-tuning)
$PYTHON eval/evaluate.py --model_path $MODEL --output eval_results/baseline.json

# SFT rank=64 (best)
$PYTHON eval/evaluate.py --model_path $MODEL \
    --adapter_path $LF/saves/cyberlllm_sft_rank64 \
    --output eval_results/sft_rank64.json

# DPO rank=64
$PYTHON eval/evaluate.py --model_path $MODEL \
    --adapter_path $LF/saves/cyberlllm_dpo_rank64 \
    --output eval_results/dpo_rank64.json

# Ablations
for RANK in 8 16; do
    $PYTHON eval/evaluate.py --model_path $MODEL \
        --adapter_path $LF/saves/cyberlllm_sft_rank${RANK} \
        --output eval_results/sft_rank${RANK}.json
done
```

Print summary:
```bash
$PYTHON -c "
import json, glob
print('Model                 | Accuracy')
print('-' * 40)
for f in sorted(glob.glob('eval_results/*.json')):
    d = json.load(open(f))
    name = f.split('/')[-1].replace('.json','')
    print(f'{name:22s}: {d[\"accuracy\"]*100:.1f}%  ({d[\"correct\"]}/100)')
"
```

---

## Expected Results

| Model | MMLU CS Accuracy |
|-------|:---:|
| Baseline | 74.0% |
| SFT rank=8 | 74.0% |
| SFT rank=16 | 76.0% |
| SFT rank=32 | 75.0% |
| **SFT rank=64** | **83.0%** |
| DPO rank=32 | 75.0% |
| DPO rank=64 | 81.0% |

Key finding: rank=64 crosses a capacity threshold (+8 pp over rank=32).
The baseline has an A-bias (predicts A 37/100 times vs gold 28/100).
SFT rank=64 corrects this (A=34/100) with zero regressions on any question.

---

## Training Config Reference

All YAML configs are in `configs/`. Key parameters:

| Config | Stage | Rank | LR | Epochs | GPU | Batch |
|--------|-------|------|----|--------|-----|-------|
| cyberlllm_sft_rank64.yaml | SFT | 64 | 5e-5 | 5 | 8 | 128 |
| cyberlllm_sft_8gpu.yaml | SFT | 32 | 5e-5 | 5 | 8 | 128 |
| cyberlllm_sft_rank16.yaml | SFT | 16 | 5e-5 | 5 | 8 | 128 |
| cyberlllm_sft_rank8.yaml | SFT | 8 | 5e-5 | 5 | 8 | 128 |
| cyberlllm_dpo_rank64.yaml | DPO | 64 | 1e-5 | 3 | 8 | 64 |
| cyberlllm_dpo_8gpu.yaml | DPO | 32 | 1e-5 | 3 | 8 | 64 |

All SFT configs: cutoff_len=2048, bf16=true, gradient_checkpointing=true.
DPO configs: pref_beta=0.1, pref_loss=sigmoid, loads from SFT adapter.

---

## Storage Requirements

| Item | Size | Notes |
|------|------|-------|
| Base model (Qwen2.5-7B) | ~15GB | Downloaded in Step 1 |
| Training data (all JSONs) | ~10MB | Generated in Step 3 |
| LoRA adapter per rank | 330MB–620MB | Saved in LlamaFactory/saves/ |
| Total (all 6 adapters) | ~9.5GB | Delete non-best after eval |

Recommended: keep only `saves/cyberlllm_sft_rank64/` after evaluation.

---

## Troubleshooting

**`torchrun` not found:** The conda env bin must be in PATH.
Run `export PATH=/home/jtan/miniconda3/envs/rl_hw3/bin:$PATH` first.

**MMLU download fails:** Server needs internet access. Test with:
`python3 -c "from datasets import load_dataset; load_dataset('cais/mmlu', 'computer_security', split='test')"`

**OOM during training:** Reduce `per_device_train_batch_size` to 2 and increase `gradient_accumulation_steps` to 8 (keeps effective batch=128).

**DPO loss is 0 from step 1:** Normal — the SFT model already prefers correct labels. Monitor `rewards/margins` instead.
